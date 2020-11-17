// GPU - PBO 이용
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

// 복소수 구조체
struct ucomplex {
    float real;
    float imag;
};

const int width = 1024; // 넓이
float scope = 0.8; // 확대,축소
float pre_scope = 0.0;
float Dx = 0.0, Dy = 0.0; // mouse move
bool TimeDir = true, Mode = true; // 확대 축소 방향 . time/scroll 
int key_s = 0; // 방정식 바꾸는 key
int key_d = 0; // 차수 늘리는 key
float Zoom = -50.0;
int StartPt[2];

//GPU 관련 변수
dim3 dimGrid;
dim3 dimBlock;
float* zD; // 뉴턴방정식 계산 결과를 저장할 배열
float p = 0;
GLuint gl_pbo; // 픽셀 버퍼를 가르키는 OpenGL 핸들 
cudaGraphicsResource* cuda_pbo; // 픽셀 버퍼를 가르키는 CUDA 핸들 
uchar4* pDevImage; // 픽셀 버퍼에 대한 실제 메모리 주소

// 콜백 함수
void Keyboard(unsigned char key, int x, int y);
void Mouse(int button, int state, int x, int y);
void MouseMove(int x, int y);
void MouseWheel(int button, int dir, int x, int y);
void Reshape(int w, int h);
void Timer(int id);
void Close();
void Display();

//사용자 정의함수
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
    // 1. 초기 방정식: x^3 - 1
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
    // 뉴턴식 계산 - 픽셀의 고유 iter값을 얻는다
    while (abs(f(x, Key_d, Key_s)) > eps&& iter <= maxiter) {
        iter++;
        x = x - f(x, Key_d, Key_s) / df(x, Key_d, Key_s); // 점화식
    }
    return iter;
}

int main(int argc, char** argv) {

    // opengl 초기화
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA);
    //glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);

    // 윈도우 설정
    glutInitWindowSize(width, width);
    //glutInitWindowSize(700, 700);
    glutInitWindowPosition(0, 0);
    glutCreateWindow("Newton");

    // 관측설정
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-2, 2, 2, -2, 1, -1);

    // 콜백함수 등록
    glutDisplayFunc(Display);
    glutTimerFunc(30, Timer, 0);
    glutReshapeFunc(Reshape);
    glutKeyboardFunc(Keyboard);
    glutMouseFunc(Mouse);
    glutMotionFunc(MouseMove);
    glutMouseWheelFunc(MouseWheel);
    glutCloseFunc(Close);

    // grid, block 만들기
    const int block_width = 16;
    cudaMalloc((void**)&zD, width * width * sizeof(float));
    dimGrid = dim3((width - 1 / block_width) + 1, (width - 1 / block_width) + 1);
    dimBlock = dim3(block_width, block_width);


    // glew 초기화
    glewInit();

    // GPU Device 세팅
    cudaSetDevice(0);
    cudaGLSetGLDevice(0);
    glGenBuffers(1, &gl_pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_pbo);

    // 픽셀 버퍼 할당
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width * width * sizeof(uchar4), NULL, GL_DYNAMIC_DRAW_ARB);

    // 이벤트 처리 루프 진입
    glutMainLoop();

    // 동적할당 해제
    cudaFree(zD);
}

void MathGPU() {

    // 화면 비율에 맞춰 확대축소 비율 계산
    if (width < 1000)
        p = 1;
    else
        p = (width / 999);
    float scopenum = p * 15 * sin(scope);

    float dx = Dx, dy = Dy;
    MathGPUKernel << <dimGrid, dimBlock >> > (zD, scopenum, dx, dy, key_d, key_s, pDevImage); // 커널함수 호출
    cudaDeviceSynchronize(); // 동기화
}

__global__ void MathGPUKernel(float* zD, float scopenum, float dx, float dy, int Key_d, int Key_s, uchar4* DevImage) {
    float xi = blockIdx.x * blockDim.x + threadIdx.x;
    float yi = blockIdx.y * blockDim.y + threadIdx.y;
    int maxiter = 255;
    float eps = 0.0001;
    // 뉴턴 점화식 계산
    uchar4 C;
    if (xi < width && yi < width) {

        // 각 픽셀별 복소수값 할당 후 뉴턴값 계산
        // 마우스 스크롤(scope) -> 전체 범위가 확대 축소(곱 연산)
        // 마우스 드래그(dx,dy) -> 범위의 크기는 그대로이고 범위가 평행이동(+연산)
        ucomplex xy = { (xi / width * 4 - 2) * scopenum + dx, (yi / width * 4 - 2) * scopenum + dy };
        zD[(int)yi * width + (int)xi] = newton(xy, eps, maxiter, Key_d, Key_s);

        // 색상 계산
        float color = zD[(int)yi * width + (int)xi] / maxiter;
        int offset = ((int)yi * width + (int)xi);
        C.x = color * 13 * 255;
        C.y = color * 33 * 255;
        C.z = color * 49 * 255;
        C.w = 255;
        DevImage[offset] = C;
    }
}

// width * width 픽셀에 색상을 지정
void Display() {

    // 배경 초기화
    __int64 _st = GetMicroSecond();
    glClearColor(1, 1, 1, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    // 픽셀 버퍼를 CUDA에 등록
    cudaGraphicsGLRegisterBuffer(&cuda_pbo, gl_pbo, cudaGraphicsMapFlagsNone);

    // 픽셀 버퍼를 CUDA 시스템에 mapping
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
    // GPU 리셋
    cudaDeviceReset();

    //픽셀 버퍼의 바인딩 해제 및 핸들 제거
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    glDeleteBuffers(1, &gl_pbo);
}

void Timer(int id)
{
    // Mode = true 일때 시간에따라 자동 확대, 축소 (key = f)
    // TimeDir = true 이면 축소 false이면 확대
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
    glutTimerFunc(30, Timer, 0); //30ms마다 Timer함수 호출
}

void Reshape(int w, int h)
{
    glViewport(0, 0, w, h);
}

void Keyboard(unsigned char key, int x, int y)
{
    if (key == 27)
        exit(-1);
    // 방정식 바꾸기
    if (key == 's')
    {
        key_d = 0;
        key_s = (key_s + 1) % 4;
    }
    // 차수 늘리기
    if (key == 'd')
    {
        key_d++;
    }
    // 차수 낮추기
    if (key == 'e')
    {
        if (key_d > 0)
            key_d--;
    }
    // 자동 확대 모드 바꾸기
    if (key == 'f') {
        Mode = !Mode;
    }
    glutPostRedisplay();
}

void Mouse(int button, int state, int x, int y)
{
    // 마우스 버튼을 누른 경우,
    if (state == GLUT_DOWN)
    {
        StartPt[0] = x;
        StartPt[1] = y;
    }

    // 마우스 버튼을 땐 경우,
    if (state == GLUT_UP)
        StartPt[0] = StartPt[1] = 0;
}

void MouseMove(int x, int y) {
    Dx += -(x - StartPt[0]) * 0.01; // x 이동값
    Dy += -(StartPt[1] - y) * 0.01; // y 이동값
}

void MouseWheel(int button, int dir, int x, int y)
{
    // Mode = false 일때
    // 수동 확대 축소
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
