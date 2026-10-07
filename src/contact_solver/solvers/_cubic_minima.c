/* Batched continuous separation kernel: same derivative partition root isolation.
 * No fast-math, no discarded small leading terms, no sampling acceptance. */
#include <math.h>
#include <float.h>
#include <stdlib.h>

static double value(const double *p, int degree, double x) {
    double y=p[degree];
    for(int i=degree-1;i>=0;--i)y=y*x+p[i];
    return y;
}
static int compare(const void *a,const void *b) {
    double x=*(const double*)a,y=*(const double*)b;
    return (x>y)-(x<y);
}
static int unique(double *a,int n) {
    qsort(a,n,sizeof(double),compare);
    int k=0;
    for(int i=0;i<n;++i)if(k==0 || a[i]!=a[k-1])a[k++]=a[i];
    return k;
}
static double bracket(const double *p,int degree,double a,double b,double fa) {
    double x=(a+b)*0.5;
    for(int k=0;k<100;++k) {
        double fx=value(p,degree,x);
        if(fx==0)return x;
        if(signbit(fx)==signbit(fa)){a=x;fa=fx;}else b=x;
        if(b-a<=1e-14+4*DBL_EPSILON*fabs(x))return (a+b)*0.5;
        double derivative=degree*p[degree];
        for(int j=degree-1;j>=1;--j)derivative=derivative*x+j*p[j];
        double trial=x-fx/derivative;
        /* Safeguard Newton by insisting on progress inside the bracket. */
        /* Every other step bisects: at least 50 halvings within the cap,
         * even for a very flat root. There is no silent loose termination. */
        x=(k%2==0&&isfinite(trial)&&trial>a+0.1*(b-a)&&trial<b-0.1*(b-a))?trial:(a+b)*0.5;
    }
    return x;
}
static int roots(const double *input,int degree,double *out) {
    while(degree>0 && input[degree]==0)--degree;
    if(degree==0)return 0;
    double p[6],largest=0;
    for(int i=0;i<=degree;++i)largest=fmax(largest,fabs(input[i]));
    for(int i=0;i<=degree;++i)p[i]=input[i]/largest;
    if(degree==1){double x=-p[0]/p[1];if(x>=0&&x<=1){out[0]=x;return 1;}return 0;}
    double derivative[5],critical[32],partition[34];
    for(int i=1;i<=degree;++i)derivative[i-1]=i*p[i];
    int count=roots(derivative,degree-1,critical),n=0;
    partition[0]=0;partition[1]=1;
    for(int i=0;i<count;++i){partition[i+2]=critical[i];out[n++]=critical[i];}
    int parts=unique(partition,count+2);
    for(int i=0;i<parts-1;++i){
        double a=partition[i],b=partition[i+1],fa=value(p,degree,a),fb=value(p,degree,b);
        if(fa==0)out[n++]=a;
        if(fb==0)out[n++]=b;
        if(fa!=0&&fb!=0&&signbit(fa)!=signbit(fb))out[n++]=bracket(p,degree,a,b,fa);
    }
    return unique(out,n);
}
static double distance(const double *c,int d,double t) {
    double sum=0;
    for(int j=0;j<d;++j){double x=c[j]+t*(c[d+j]+t*(c[2*d+j]+t*c[3*d+j]));sum+=x*x;}
    return sqrt(sum);
}
static double lower_bound(const double *c,int d,double *guard) {
    double controls[4*d],direction[d],delta[d],sq=0,dot=0,magnitude=0;
    for(int j=0;j<d;++j){
        controls[j]=c[j];controls[d+j]=c[j]+c[d+j]/3;
        controls[2*d+j]=c[j]+2*c[d+j]/3+c[2*d+j]/3;
        controls[3*d+j]=c[j]+c[d+j]+c[2*d+j]+c[3*d+j];
        delta[j]=controls[3*d+j]-c[j];sq+=delta[j]*delta[j];dot+=c[j]*delta[j];
        for(int k=0;k<4;++k)magnitude+=fabs(c[k*d+j]);
    }
    double u=sq>0?fmax(0,fmin(1,-dot/sq)):0,norm=0,box=0;
    for(int j=0;j<d;++j){
        double lo=controls[j],hi=lo;
        for(int k=1;k<4;++k){lo=fmin(lo,controls[k*d+j]);hi=fmax(hi,controls[k*d+j]);}
        double near=fmax(0,fmax(lo,-hi));box+=near*near;
        direction[j]=c[j]+u*delta[j];norm+=direction[j]*direction[j];
    }
    norm=sqrt(norm);double support=INFINITY;
    if(norm>0){for(int k=0;k<4;++k){double s=0;for(int j=0;j<d;++j)s+=controls[k*d+j]*(direction[j]/norm);support=fmin(support,s);}}
    else support=0;
    *guard=1024*DBL_EPSILON*fmax(1,magnitude);
    return fmax(0,fmax(sqrt(box),support)-*guard);
}
/* Outputs per-polynomial minima. Pruned entries stay +inf. */
int batch_minima(const double *coefficients,int count,int d,int pruning,double *times,double *distances,int *solved) {
    double upper=INFINITY,best=INFINITY;
    *solved=0;
    if(pruning)for(int i=0;i<count;++i){
        const double *c=coefficients+i*4*d;double magnitude=0;
        for(int j=0;j<d;++j)for(int k=0;k<4;++k)magnitude+=fabs(c[k*d+j]);
        double guard=1024*DBL_EPSILON*fmax(1,magnitude);
        upper=fmin(upper,fmin(distance(c,d,0),fmin(distance(c,d,.5),distance(c,d,1)))+guard);
    }
    for(int i=0;i<count;++i){
        const double *c=coefficients+i*4*d;double guard=0;
        distances[i]=INFINITY;times[i]=0;
        if(pruning && lower_bound(c,d,&guard)>fmin(upper,best)+guard)continue;
        ++*solved;
        double p[6]={0},candidates[64];
        for(int j=0;j<d;++j)for(int k=0;k<4;++k)for(int l=0;l<3;++l)p[k+l]+=c[k*d+j]*(l+1)*c[(l+1)*d+j];
        for(int k=0;k<6;++k)if(!isfinite(p[k]))return -1;
        int nr=roots(p,5,candidates);
        candidates[nr++]=0;candidates[nr++]=1;nr=unique(candidates,nr);
        for(int k=0;k<nr;++k){double q=distance(c,d,candidates[k]);if(q<distances[i]){times[i]=candidates[k];distances[i]=q;}}
        best=fmin(best,distances[i]);
    }
    return 0;
}

/* Python buffer API keeps the build independent of NumPy headers. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h>

static int double_buffer(PyObject *object, Py_buffer *buffer, int writable) {
    int flags = PyBUF_FORMAT | PyBUF_C_CONTIGUOUS;
    if (writable) flags |= PyBUF_WRITABLE;
    if (PyObject_GetBuffer(object, buffer, flags) < 0) return -1;
    if (buffer->itemsize != sizeof(double) || strcmp(buffer->format, "d") != 0) {
        PyErr_SetString(PyExc_ValueError, "Expected contiguous native float64 buffers");
        return -1;
    }
    return 0;
}

static PyObject *python_minima(PyObject *self, PyObject *args) {
    PyObject *coefficients, *times, *distances;
    Py_buffer c = {0}, t = {0}, r = {0};
    int count, dimension, pruning, solved, status;
    if (!PyArg_ParseTuple(args, "OiipOO", &coefficients, &count, &dimension,
                          &pruning, &times, &distances)) return NULL;
    if (count < 0 || dimension < 1 ||
        (Py_ssize_t)count > PY_SSIZE_T_MAX / (Py_ssize_t)sizeof(double) / 4 / dimension) {
        PyErr_SetString(PyExc_ValueError, "Invalid batch dimensions");
        return NULL;
    }
    if (double_buffer(coefficients, &c, 0) < 0 || double_buffer(times, &t, 1) < 0 ||
        double_buffer(distances, &r, 1) < 0) goto error;
    if (c.len != (Py_ssize_t)count * 4 * dimension * (Py_ssize_t)sizeof(double) ||
        t.len != (Py_ssize_t)count * (Py_ssize_t)sizeof(double) || r.len != t.len) {
        PyErr_SetString(PyExc_ValueError, "Buffer sizes do not match batch dimensions");
        goto error;
    }
    status = batch_minima(c.buf, count, dimension, pruning, t.buf, r.buf, &solved);
    PyBuffer_Release(&c); PyBuffer_Release(&t); PyBuffer_Release(&r);
    if (status) {
        PyErr_SetString(PyExc_ValueError, "Nonfinite distance derivative polynomial");
        return NULL;
    }
    return PyLong_FromLong(solved);
error:
    if (c.obj) PyBuffer_Release(&c);
    if (t.obj) PyBuffer_Release(&t);
    if (r.obj) PyBuffer_Release(&r);
    return NULL;
}

static PyMethodDef methods[] = {
    {"minima", python_minima, METH_VARARGS, "Compute batched cubic distance minima."},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_cubic_minima", NULL, -1, methods};
PyMODINIT_FUNC PyInit__cubic_minima(void) { return PyModule_Create(&module); }
