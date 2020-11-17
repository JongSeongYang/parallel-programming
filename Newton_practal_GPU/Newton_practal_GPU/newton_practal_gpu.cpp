// GPU - PBO �̿�
#include <math.h>
//#include <glut.h>
#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include <iostream>
#include <complex>
#include <cuComplex.h>
#include <thrust/complex.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "..\usr\include\GL\glew.h"
#include "..\usr\include\GL\freeglut.h"
#include "cuda_gl_interop.h"
using namespace std;

// ���Ҽ� ����ü
struct ucomplex {
    float real;
    float imag;
};

const int width = 1024; // ����
float scope = 0.8; // Ȯ��,���
float pre_scope = 0.0;
float Dx = 0.0, Dy = 0.0; // mouse move
bool TimeDir = true, Mode = true; // Ȯ�� ��� ���� . time/scroll 
int key_s = 0; // ������ �ٲٴ� key
int key_d = 0; // ���� �ø��� key
float Zoom = -50.0;
int StartPt[2];

//GPU ���� ����
dim3 dimGrid;
dim3 dimBlock;
float* zD; // ���Ϲ����� ��� ����� ������ �迭
float p = 0;
GLuint gl_pbo; // �ȼ� ���۸� ����Ű�� OpenGL �ڵ� 
cudaGraphicsResource* cuda_pbo; // �ȼ� ���۸� ����Ű�� CUDA �ڵ� 
uchar4* pDevImage; // �ȼ� ���ۿ� ���� ���� �޸� �ּ�

// �ݹ� �Լ�
void Keyboard(unsigned char key, int x, int y);
void Mouse(int button, int state, int x, int y);
void MouseMove(int x, int y);
void MouseWheel(int button, int dir, int x, int y);
void Reshape(int w, int h);
void Timer(int id);
void Close();
void Display();

//����� �����Լ�
__global__ void MathGPUKernel(float* zD, float scopenum, float dx, float dy, int Key_d, int Key_s, uchar4* DevImage);
__int64 GetMicroSecond();
__host__ __device__ float abs(ucomplex x) {
    return sqrt(x.real * x.real + x.imag * x.imag);
}
__host__ __device__ ucomplex operator*(ucomplex x, ucomplex y) {
    ucomplex c = {
    (x.real * y.real - x.imag * y.imag),
    (x.imag * y.real + x.real * y.imag)
    };
    return c;
}
__host__ __device__ ucomplex operator*(ucomplex x, float dy) {
    ucomplex y = { dy, 0.0 };
    return x * y;
}
__host__ __device__ ucomplex operator/(ucomplex x, ucomplex y) {
    ucomplex c = {
    (x.real * y.real + x.imag * y.imag) / (y.real * y.real + y.imag * y.imag),
    (x.imag * y.real - x.real * y.imag) / (y.real * y.real + y.imag * y.imag)
    };
    return c;
}
__host__ __device__ ucomplex operator+(ucomplex x, ucomplex y) {
    ucomplex c = {
    (x.real + y.real),
    (x.imag + y.imag)
    };
    return c;
}
__host__ __device__ ucomplex operator+(ucomplex x, float dy) {
    ucomplex y = { dy, 0.0 };
    return x + y;
}
__host__ __device__ ucomplex operator-(ucomplex x, ucomplex y) {
    ucomplex c = {
    (x.real - y.real),
    (x.imag - y.imag)
    };
    return c;
}
__host__ __device__ ucomplex operator-(ucomplex x, float dy) {
    ucomplex y = { dy, 0.0 };
    return x - y;
}
__host__ __device__ ucomplex f(ucomplex x, int Key_d, int Key_s) {
    // 1. �ʱ� ������: x^3 - 1
    ucomplex d = x * x * x - 1.0;
    if (Key_s == 0)
    {
        ucomplex temp_d = x * x * x;
        if (Key_d > 0) {
            for (int i = 0; i < Key_d; i++)
                temp_d = temp_d * x;
        }
        d = temp_d - 1.0;
    }
    // 2. x^3 - 2x^2 + 2
    else if (Key_s == 1)
    {
        d = x * x * x - x * x * 2.0 + 2.0;
    }
    // 3. x^8 + 15x^4 - 16
    else if (Key_s == 2)
    {
        d = x * x * x * x * x * x * x * x + x * x * x * x * 15.0 - 16.0;
    }
    // 4. sin(x)
    else if (Key_s == 3)
    {
        thrust::complex<float> c = thrust::complex<float>(x.real, x.imag);
        d = {
           sin(c).real(),
           sin(c).imag()
        };
    }
    return d;
}
__host__ __device__ ucomplex df(ucomplex x, int Key_d, int Key_s) {
    ucomplex d = x * x * 3;
    // 1. 3x^2
    if (Key_s == 0)
    {
        ucomplex temp_d = x * x;
        float de = 3.0;
        if (Key_d > 0) {
            for (int i = 0; i < Key_d; i++) {
                temp_d = temp_d * x;
                de = de + 1.0;
            }
        }
        d = temp_d * de;

    }
    // 2. 3x^2 - 4x
    else if (Key_s == 1)
    {
        d = x * x * 3 - x * 4.0;
    }
    // 3. 8x^7 +60x^3
    else if (Key_s == 2)
    {
        d = x * x * x * x * x * x * x * 8.0 + x * x * x * 60.0;
    }
    // 4. cos(x)
    else if (Key_s == 3)
    {
        thrust::complex<float> c = thrust::complex<float>(x.real, x.imag);
        d = {
           cos(c).real(),
           cos(c).imag()
        };
    }
    return d;
}
__host__ __device__ float newton(ucomplex x0, float eps, int maxiter, int Key_d, int Key_s) {
    ucomplex x = x0;
    int iter = 0;
    // ���Ͻ� ��� - �ȼ��� ���� iter���� ��´�
    while (abs(f(x, Key_d, Key_s)) > eps&& iter <= maxiter) {
        iter++;
        x = x - f(x, Key_d, Key_s) / df(x, Key_d, Key_s); // ��ȭ��
    }
    return iter;
}

int main(int argc, char** argv) {

    // opengl �ʱ�ȭ
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA);
    //glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);

    // ������ ����
    glutInitWindowSize(width, width);
    //glutInitWindowSize(700, 700);
    glutInitWindowPosition(0, 0);
    glutCreateWindow("Newton");

    // ��������
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-2, 2, 2, -2, 1, -1);

    // �ݹ��Լ� ���
    glutDisplayFunc(Display);
    glutTimerFunc(30, Timer, 0);
    glutReshapeFunc(Reshape);
    glutKeyboardFunc(Keyboard);
    glutMouseFunc(Mouse);
    glutMotionFunc(MouseMove);
    glutMouseWheelFunc(MouseWheel);
    glutCloseFunc(Close);

    // grid, block �����
    const int block_width = 16;
    cudaMalloc((void**)&zD, width * width * sizeof(float));
    dimGrid = dim3((width - 1 / block_width) + 1, (width - 1 / block_width) + 1);
    dimBlock = dim3(block_width, block_width);


    // glew �ʱ�ȭ
    glewInit();

    // GPU Device ����
    cudaSetDevice(0);
    cudaGLSetGLDevice(0);
    glGenBuffers(1, &gl_pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_pbo);

    // �ȼ� ���� �Ҵ�
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width * width * sizeof(uchar4), NULL, GL_DYNAMIC_DRAW_ARB);

    // �̺�Ʈ ó�� ���� ����
    glutMainLoop();

    // �����Ҵ� ����
    cudaFree(zD);
}

void MathGPU() {

    // ȭ�� ������ ���� Ȯ����� ���� ���
    if (width < 1000)
        p = 1;
    else
        p = (width / 999);
    float scopenum = p * 15 * sin(scope);

    float dx = Dx, dy = Dy;
    MathGPUKernel << <dimGrid, dimBlock >> > (zD, scopenum, dx, dy, key_d, key_s, pDevImage); // Ŀ���Լ� ȣ��
    cudaDeviceSynchronize(); // ����ȭ
}

__global__ void MathGPUKernel(float* zD, float scopenum, float dx, float dy, int Key_d, int Key_s, uchar4* DevImage) {
    float xi = blockIdx.x * blockDim.x + threadIdx.x;
    float yi = blockIdx.y * blockDim.y + threadIdx.y;
    int maxiter = 255;
    float eps = 0.0001;
    // ���� ��ȭ�� ���
    uchar4 C;
    if (xi < width && yi < width) {

        // �� �ȼ��� ���Ҽ��� �Ҵ� �� ���ϰ� ���
        // ���콺 ��ũ��(scope) -> ��ü ������ Ȯ�� ���(�� ����)
        // ���콺 �巡��(dx,dy) -> ������ ũ��� �״���̰� ������ �����̵�(+����)
        ucomplex xy = { (xi / width * 4 - 2) * scopenum + dx, (yi / width * 4 - 2) * scopenum + dy };
        zD[(int)yi * width + (int)xi] = newton(xy, eps, maxiter, Key_d, Key_s);

        // ���� ���
        float color = zD[(int)yi * width + (int)xi] / maxiter;
        int offset = ((int)yi * width + (int)xi);
        C.x = color * 13 * 255;
        C.y = color * 33 * 255;
        C.z = color * 49 * 255;
        C.w = 255;
        DevImage[offset] = C;
    }
}

// width * width �ȼ��� ������ ����
void Display() {

    // ��� �ʱ�ȭ
    __int64 _st = GetMicroSecond();
    glClearColor(1, 1, 1, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    // �ȼ� ���۸� CUDA�� ���
    cudaGraphicsGLRegisterBuffer(&cuda_pbo, gl_pbo, cudaGraphicsMapFlagsNone);

    // �ȼ� ���۸� CUDA �ý��ۿ� mapping
    size_t size;
    cudaGraphicsMapResources(1, &cuda_pbo, NULL);
    cudaGraphicsResourceGetMappedPointer((void**)&pDevImage, &size, cuda_pbo);

    MathGPU();
    glDrawPixels(width, width, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    printf("Elapsed time: %I64d mico sec \n", GetMicroSecond() - _st);
    glFinish();

    cudaGraphicsUnmapResources(1, &cuda_pbo, NULL);
    cudaGraphicsUnregisterResource(cuda_pbo);
}

void Close() {
    // GPU ����
    cudaDeviceReset();

    //�ȼ� ������ ���ε� ���� �� �ڵ� ����
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    glDeleteBuffers(1, &gl_pbo);
}

void Timer(int id)
{
    // Mode = true �϶� �ð������� �ڵ� Ȯ��, ��� (key = f)
    // TimeDir = true �̸� ��� false�̸� Ȯ��
    if (Mode) {
        if (TimeDir == true) {
            scope -= 0.05;
            if (scope <= 0.00001) {
                scope = 0.00001;
                TimeDir = false;
            }
        }
        else {
            scope += 0.05;
            if (scope >= 1.57) {
                scope = 1.57;
                TimeDir = true;
            }
        }
    }
    glutPostRedisplay();
    glutTimerFunc(30, Timer, 0); //30ms���� Timer�Լ� ȣ��
}

void Reshape(int w, int h)
{
    glViewport(0, 0, w, h);
}

void Keyboard(unsigned char key, int x, int y)
{
    if (key == 27)
        exit(-1);
    // ������ �ٲٱ�
    if (key == 's')
    {
        key_d = 0;
        key_s = (key_s + 1) % 4;
    }
    // ���� �ø���
    if (key == 'd')
    {
        key_d++;
    }
    // ���� ���߱�
    if (key == 'e')
    {
        if (key_d > 0)
            key_d--;
    }
    // �ڵ� Ȯ�� ��� �ٲٱ�
    if (key == 'f') {
        Mode = !Mode;
    }
    glutPostRedisplay();
}

void Mouse(int button, int state, int x, int y)
{
    // ���콺 ��ư�� ���� ���,
    if (state == GLUT_DOWN)
    {
        StartPt[0] = x;
        StartPt[1] = y;
    }

    // ���콺 ��ư�� �� ���,
    if (state == GLUT_UP)
        StartPt[0] = StartPt[1] = 0;
}

void MouseMove(int x, int y) {
    Dx += -(x - StartPt[0]) * 0.01; // x �̵���
    Dy += -(StartPt[1] - y) * 0.01; // y �̵���
}

void MouseWheel(int button, int dir, int x, int y)
{
    // Mode = false �϶�
    // ���� Ȯ�� ���
    if (!Mode) {
        if (dir > 0)
        {
            pre_scope = scope;
            scope -= 0.01;
            if (scope <= 0.00001)
                scope = 0.00001;
        }
        else
        {
            pre_scope = scope;
            scope += 0.01;
            if (scope >= 1.57)
                scope = 1.57;
        }
    }
    glutPostRedisplay();
}

__int64 GetMicroSecond()
{
    LARGE_INTEGER frequency;
    LARGE_INTEGER now;

    if (!QueryPerformanceFrequency(&frequency))
        return (__int64)GetTickCount();

    if (!QueryPerformanceCounter(&now))
        return (__int64)GetTickCount();

    return((now.QuadPart) / (frequency.QuadPart / 1000000));
}
