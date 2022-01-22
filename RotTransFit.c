/** Fit a set of points to a model that is subject to rotation and translation.
 *
 * The model is defined by a set of primitive faces such as plane, cylinder, sphere, etc.
 * The measured points are first translated by (xt, yt, zt),
 * then rotated about x-y-z axes by g, b, a angles in sequence,
 *
 * These 6 parameters are the output of the fit.
 *
 * Author: Yuan Mei
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multifit_nlinear.h>

#ifdef LINE_MAX
#undef LINE_MAX
#define LINE_MAX 4096
#endif

#ifndef LINE_MAX
#define LINE_MAX 4096
#endif

/** Check if the character is a field separator. */
#ifndef sepq
#define sepq(a) ((a)==' ' || (a)=='\t')
#endif

#define NUM_PENALTY 1000000 /**< numerical penalty, assigned to values
                                 when faulty conditions occur */
#define NDIM        3       /**< number of dimensions */

enum {
    FACE_PLANE=1,
    FACE_CYLINDER=2,
    FACE_SPHERE=3,
    FACE_LINE=12,
    FACE_POINT=13
};

struct face {
    int ftype; /**< type of face: plane, cylinder etc... */
    double x0; /**< origin */
    double y0;
    double z0;
    double nx; /**< normal vector */
    double ny;
    double nz;
    double r;  /**< radius */
};

struct data {
    double pr;   /**< probe radius */
    const struct face *faces;
    size_t n;    /**< number of measured data points */
    size_t *fid; /**< idx of face in the registered faces */
    double *x;   /**< measured point */
    double *y;
    double *z;
    double *w;   /**< weights */
    double *x1;  /**< after transformation */
    double *y1;
    double *z1;
};

/** Distance from measured point to face. */
int dist_f(const gsl_vector *p, void *data, gsl_vector *f)
{
    double pr = ((struct data *)data)->pr;
    const struct face *faces = ((struct data *)data)->faces;
    size_t n = ((struct data *)data)->n;
    size_t *fid = ((struct data *)data)->fid;
    double *x = ((struct data *)data)->x;
    double *y = ((struct data *)data)->y;
    double *z = ((struct data *)data)->z;

    double a  = gsl_vector_get(p, 0), Ca = cos(a), Sa = sin(a);
    double b  = gsl_vector_get(p, 1), Cb = cos(b), Sb = sin(b);
    double g  = gsl_vector_get(p, 2), Cg = cos(g), Sg = sin(g);
    double xt = gsl_vector_get(p, 3);
    double yt = gsl_vector_get(p, 4);
    double zt = gsl_vector_get(p, 5);

    size_t i;
    for (i = 0; i < n; i++) {
        int ftype = faces[fid[i]].ftype;
        double x0 = faces[fid[i]].x0;
        double y0 = faces[fid[i]].y0;
        double z0 = faces[fid[i]].z0;
        double nx = faces[fid[i]].nx;
        double ny = faces[fid[i]].ny;
        double nz = faces[fid[i]].nz;
        double r  = faces[fid[i]].r;

        double x1, y1, z1;
        x1 = Ca*Cb * (x[i]+xt) + (Ca*Sb*Sg-Sa*Cg) * (y[i]+yt) + (Ca*Sb*Cg+Sa*Sg) * (z[i]+zt);
        y1 = Sa*Cb * (x[i]+xt) + (Sa*Sb*Sg+Ca*Cg) * (y[i]+yt) + (Sa*Sb*Cg-Ca*Sg) * (z[i]+zt);
        z1 =  - Sb * (x[i]+xt) +            Cb*Sg * (y[i]+yt) +            Cb*Cg * (z[i]+zt);

        ((struct data *)data)->x1[i] = x1;
        ((struct data *)data)->y1[i] = y1;
        ((struct data *)data)->z1[i] = z1;

        double d0, d1, dist=0.0;
        switch (abs(ftype)) {
        case FACE_PLANE:
            d0 = -(nx*x0 + ny*y0 + nz*z0);
            d1 = (nx*x1 + ny*y1 + nz*z1);
            dist = (d1 + d0) / sqrt(nx*nx + ny*ny + nz*nz);
            /* if dist<0, the point is on the wrong side of the plane.
             * The next line will penalize this condition. */
            dist -= pr;
            break;
        case FACE_CYLINDER:
        case FACE_LINE:
        {
            double dx = x1 - x0, dy = y1 - y0, dz = z1 - z0;
            double s1 = dy * nz - dz * ny, s2 = dz*nx - dx*nz, s3 = dx*ny - dy*nx;
            dist = sqrt((s1*s1 + s2*s2 + s3*s3)/(nx*nx + ny*ny + nz*nz)) - r;
            if (abs(ftype) == FACE_CYLINDER) {
                if (ftype > 0) { /* point outside of the cylinder */
                    dist -= pr;
                } else { /* inside */
                    dist = -dist - pr;
                }
            }
        }
            break;
        case FACE_SPHERE:
        case FACE_POINT:
            dist = sqrt((x1-x0)*(x1-x0) + (y1-y0)*(y1-y0) + (z1-z0)*(z1-z0)) - r;
            if (abs(ftype) == FACE_SPHERE) {
                if (ftype > 0) { /* point outside of the sphere */
                    dist -= pr;
                } else { /* inside */
                    dist = -dist - pr;
                }
            }
            break;
        default:
            dist = 0.0;
            break;
        }
        gsl_vector_set(f, i, dist);
    }

    return GSL_SUCCESS;
}

void callback(const size_t iter, void *params,
              const gsl_multifit_nlinear_workspace *w)
{
    gsl_vector *f = gsl_multifit_nlinear_residual(w);
    gsl_vector *x = gsl_multifit_nlinear_position(w);
    double rcond=0;

    /* compute reciprocal condition number of J(x) */
    /* cond(J) = %8.4f, 1.0 / rcond */
    gsl_multifit_nlinear_rcond(&rcond, w);

    fprintf(stderr, "iter %2zu: a = %7.4f, b = %7.4f, g = %7.4f, xt = %7.4f, yt = %7.4f, zt = %7.4f, |f(x)| = %.4f\n",
            iter,
            gsl_vector_get(x, 0),
            gsl_vector_get(x, 1),
            gsl_vector_get(x, 2),
            gsl_vector_get(x, 3),
            gsl_vector_get(x, 4),
            gsl_vector_get(x, 5),
            gsl_blas_dnrm2(f));
}

/** Read a long line from file.
 * @param[inout] s string of the line, is allocated when s==NULL and n==0 and grown as needed.
 * @param[inout] n current size of s
 * @return s
 */
static char *file_read_long_line(char **s, size_t *n, FILE *fp)
{
    const int bufsz = LINE_MAX;
    char *p;
    size_t cnt, sz;

    if ( *s == NULL && *n == 0 ) {
        *n = bufsz;
        if ( (*s = calloc(*n, sizeof(char))) == NULL ) exit(-1);
    }
    p = *s;
    sz = *n;
    while ( 1 ) {
        if ( fgets(p, sz, fp) == NULL ) return NULL;
        cnt = strlen(*s);
        if ( (*s)[cnt-1] == '\n' ) {
            break;
        } else { /* line too long, expand the buffer */
            *n += bufsz;
            if ( (*s = realloc(*s, (*n)*sizeof(char))) == NULL ) exit(-1);
            p = *s + cnt;
            sz = bufsz;
        }
    }

    return *s;
}

/** Read face definition file.
 *
 * @param[inout] n number of faces read from file.  If the given value
 * *n > 0, *n is interpreted as the number of elements in the faces
 * array and only up to *n faces will be read from file.  However, if
 * the file contains less than *n faces, *n will be updated to reflect
 * the available number of elements.
 *
 * @param[inout] faces array of faces.  If *faces != NULL, *faces will
 * be used rather than allocated.
 */
int read_faces(const char *fname, size_t *n, struct face **faces)
{
    char *linebuf = NULL;
    size_t linen = 0;
    FILE *fp;
    if ((fp = fopen(fname, "r"))==NULL) {
        perror(fname);
        return -1;
    }
    struct face fc;
    int fid;
    size_t lid = 0;
    ssize_t nelem = -1;
    if (*faces == NULL) {
        /* get number of elements in the file */
        while (file_read_long_line(&linebuf, &linen, fp)) {
            lid++;
            if (linebuf[0] == '#' || linebuf[0] == '\r' || linebuf[0] == '\n') continue;
            int ret = sscanf(linebuf, "%d %d %lf %lf %lf %lf %lf %lf %lf", &fid, &fc.ftype,
                             &fc.x0, &fc.y0, &fc.z0, &fc.nx, &fc.ny, &fc.nz, &fc.r);
            if (ret < 9 || fid < 0 || fc.ftype <= 0) {
                fprintf(stderr, "Malformatted face at line %zd\n", lid);
            } else {
                if (fid > nelem) nelem = fid;
            }
        }
        nelem++;
        if (nelem == 0) {
            fprintf(stderr, "No valid face in file.\n");
            return -1;
        } else {
            fprintf(stderr, "%zd faces available in file.\n", nelem);
        }
        *n = nelem;
        if ((*faces = calloc(nelem, sizeof(struct face))) == NULL) {
            perror("calloc *faces");
            return -1;
        }
    }
    rewind(fp);
    nelem = 0;
    lid = 0;
    while (file_read_long_line(&linebuf, &linen, fp) && (nelem < *n)) {
        lid++;
        if (linebuf[0] == '#' || linebuf[0] == '\r' || linebuf[0] == '\n') continue;
        int ret = sscanf(linebuf, "%d %d %lf %lf %lf %lf %lf %lf %lf", &fid, &fc.ftype,
                         &fc.x0, &fc.y0, &fc.z0, &fc.nx, &fc.ny, &fc.nz, &fc.r);
        if (ret < 9 || fid < 0 || fc.ftype <= 0) {
            fprintf(stderr, "Malformatted face at line %zd, skipped.\n", lid);
        } else {
            if (fid >= *n) break;
            struct face *fc1 = *faces;
            memcpy(&fc1[fid], &fc, sizeof(fc));
            nelem++;
        }
    }

    fprintf(stderr, "%zd faces constructed.\n", nelem);
    free(linebuf);
    fclose(fp);
    return 0;
}

int read_points(const char *fname, size_t *n, struct data *data)
{
    char *linebuf = NULL;
    size_t linen = 0;
    FILE *fp;
    if ((fp = fopen(fname, "r"))==NULL) {
        perror(fname);
        return -1;
    }

    int fidmax = -1;
    *n = 0;
    size_t lid = 0;
    while (file_read_long_line(&linebuf, &linen, fp)) {
        lid++;
        if (linebuf[0] == '#' || linebuf[0] == '\r' || linebuf[0] == '\n') continue;
        int ret = 0;
        for (int i=0; i<linen; i++) { /* count number of ';' */
            if (linebuf[i] == ';') ret++;
        }
        int fid;
        sscanf(linebuf, "%*d;%d", &fid);
        if (ret < 8 || fid < 0) {
            fprintf(stderr, "Malformatted point at line %zd\n", lid);
        } else {
            (*n)++;
            if (fid > fidmax) fidmax = fid;
        }
    }
    fprintf(stderr, "%zd points in file.  fidmax = %d\n", *n, fidmax);
    data->n = *n;
    data->fid = calloc(data->n, sizeof(size_t));
    data->x   = calloc(data->n, sizeof(double));
    data->y   = calloc(data->n, sizeof(double));
    data->z   = calloc(data->n, sizeof(double));
    data->w   = calloc(data->n, sizeof(double));
    data->x1  = calloc(data->n, sizeof(double));
    data->y1  = calloc(data->n, sizeof(double));
    data->z1  = calloc(data->n, sizeof(double));

    rewind(fp);

    lid = 0;
    size_t idx = 0;
    while (file_read_long_line(&linebuf, &linen, fp)) {
        lid++;
        if (linebuf[0] == '#' || linebuf[0] == '\r' || linebuf[0] == '\n') continue;
        int fid = 0, ret = 0;
        double x, y, z, s;
        // ret = sscanf(linebuf, "%*d;%d;%*s ;%*d;%lf;%lf;%lf;;%lf", &fid, &x, &y, &z, &s);
        char *buf = linebuf, *endptr;
        for (; *buf != ';' ; buf++){;} buf++; /* find the next character past a ';' */
        fid = strtol(buf, &endptr, 0); if (endptr > buf) ret++; buf = endptr++;
        for (; *buf != ';' ; buf++){;} buf++;
        for (; *buf != ';' ; buf++){;} buf++;
        for (; *buf != ';' ; buf++){;} buf++;
        x = strtod(buf, &endptr); if (endptr > buf) ret++; buf = endptr++;
        for (; *buf != ';' ; buf++){;} buf++;
        y = strtod(buf, &endptr); if (endptr > buf) ret++; buf = endptr++;
        for (; *buf != ';' ; buf++){;} buf++;
        z = strtod(buf, &endptr); if (endptr > buf) ret++; buf = endptr++;
        for (; *buf != ';' ; buf++){;} buf++;
        for (; *buf != ';' ; buf++){;} buf++;
        s = strtod(buf, &endptr); if (endptr > buf) ret++; buf = endptr++;

        if (ret < 5 || fid < 0) {
            fprintf(stderr, "Malformatted point at line %zd, skipped.\n", lid);
        } else {
            data->fid[idx] = fid;
            data->x[idx] = x;
            data->y[idx] = y;
            data->z[idx] = z;
            data->w[idx] = (fabs(s)>1e-8) ? (1.0/(s*s)) : 1.0;
            idx++;
        }
    }

    fprintf(stderr, "%zd points read.\n", idx);
    free(linebuf);
    fclose(fp);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc != 10) {
        fprintf(stderr,
                "Usage: %s faces_file points_file pr a b g xt yt zt\n\n"
                "    pr is the probe radius.\n"
                "    The last 6 parameters are initial guesses.\n"
                "    Points are first translated, then rotated about x(g), y(b), z(a).\n"
                "    The 2nd column in points_file shall be faceid,\n"
                "        which is the first column of faces_file.\n"
                "    The last column in points_file shall be measurement sigma.\n", argv[0]);
        return EXIT_FAILURE;
    }
    size_t nfaces=0;
    struct face *faces=NULL;
    read_faces(argv[1], &nfaces, &faces);

    size_t nd=0;
    struct data data;
    read_points(argv[2], &nd, &data);
    data.faces = faces;

    data.pr = atof(argv[3]);

    #define np 6
    double p_init[np] = {0}; /* starting values */
    for (int i=0; i<np; i++) {
        p_init[i] = atof(argv[4+i]);
    }

    const gsl_multifit_nlinear_type *T = gsl_multifit_nlinear_trust;
    gsl_multifit_nlinear_workspace *w;
    gsl_multifit_nlinear_fdf fdf;
    gsl_multifit_nlinear_parameters fdf_params =
        gsl_multifit_nlinear_default_parameters();

    gsl_vector *f;
    gsl_matrix *J;
    gsl_matrix *covar = gsl_matrix_alloc(np, np);

    gsl_vector_view params = gsl_vector_view_array(p_init, np);
    gsl_vector_view wts = gsl_vector_view_array(data.w, nd);
    gsl_rng *r;
    double chisq, chisq0;
    int status, info;

    const double xtol = 1e-8;
    const double gtol = 1e-8;
    const double ftol = 1e-8;

    gsl_rng_env_setup();
    r = gsl_rng_alloc(gsl_rng_default);

    /* define the function to be minimized */
    fdf.f = dist_f;
    fdf.df = NULL;   /* set to NULL for finite-difference Jacobian */
    fdf.fvv = NULL;  /* not using geodesic acceleration */
    fdf.n = nd;
    fdf.p = np;
    fdf.params = &data; /* `data' supplied to the penalty function */

    /* allocate workspace with default parameters */
    w = gsl_multifit_nlinear_alloc(T, &fdf_params, nd, np);

    /* initialize solver with starting point and weights */
    gsl_multifit_nlinear_winit(&params.vector, &wts.vector, &fdf, w);

    /* compute initial cost function */
    f = gsl_multifit_nlinear_residual(w);
    gsl_blas_ddot(f, f, &chisq0);

    /* solve the system with a maximum of 100 iterations */
    status = gsl_multifit_nlinear_driver(100, xtol, gtol, ftol,
                                         callback, NULL, &info, w);

    /* compute covariance of best fit parameters */
    J = gsl_multifit_nlinear_jac(w);
    gsl_multifit_nlinear_covar(J, 0.0, covar);

    /* compute final cost */
    gsl_blas_ddot(f, f, &chisq);

#define FIT(i) gsl_vector_get(w->x, i)
#define ERR(i) sqrt(gsl_matrix_get(covar,i,i))

    fprintf(stderr, "summary from method '%s/%s'\n",
            gsl_multifit_nlinear_name(w),
            gsl_multifit_nlinear_trs_name(w));
    fprintf(stderr, "number of iterations: %zu\n",
            gsl_multifit_nlinear_niter(w));
    fprintf(stderr, "function evaluations: %zu\n", fdf.nevalf);
    fprintf(stderr, "Jacobian evaluations: %zu\n", fdf.nevaldf);
    fprintf(stderr, "reason for stopping: %s\n",
            (info == 1) ? "small step size" : "small gradient");
    fprintf(stderr, "initial |f(x)| = %f\n", sqrt(chisq0));
    fprintf(stderr, "final   |f(x)| = %f\n", sqrt(chisq));

    {
        double dof = nd - np;
        double c = GSL_MAX_DBL(1, sqrt(chisq / dof));

        fprintf(stderr, "chisq/dof = %g\n", chisq / dof);

        fprintf(stderr, "a  = %16g +/- %g\n", FIT(0), c*ERR(0));
        fprintf(stderr, "b  = %16g +/- %g\n", FIT(1), c*ERR(1));
        fprintf(stderr, "g  = %16g +/- %g\n", FIT(2), c*ERR(2));
        fprintf(stderr, "xt = %16g +/- %g\n", FIT(3), c*ERR(3));
        fprintf(stderr, "yt = %16g +/- %g\n", FIT(4), c*ERR(4));
        fprintf(stderr, "zt = %16g +/- %g\n", FIT(5), c*ERR(5));
    }

    fprintf(stderr, "status = %s\n", gsl_strerror(status));

    /* compute residual of every point */
    dist_f(w->x, &data, f); // pure distance, no weights.
    printf("#    fid   ftype  distance          x1          y1          z1\n");
    for (int i=0; i<nd; i++) {
        printf("%8zd %7d %9.6f %11.5f %11.5f %11.5f\n", data.fid[i], faces[data.fid[i]].ftype,
               gsl_vector_get(f, i), data.x1[i], data.y1[i], data.z1[i]);
    }

    gsl_multifit_nlinear_free(w);
    gsl_matrix_free(covar);
    gsl_rng_free(r);

    free(faces);
    free(data.fid);
    free(data.x);
    free(data.y);
    free(data.z);
    free(data.w);
    free(data.x1);
    free(data.y1);
    free(data.z1);
    return EXIT_SUCCESS;
}
