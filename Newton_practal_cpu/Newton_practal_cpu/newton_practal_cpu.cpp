// CPU(점&픽셀) 최종
#include "..\usr\include\GL\freeglut.h"
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <complex>
#include <memory.h>

struct ucomplex {
    float real;
    float imag;
};

const int width = 1024;
float scope = 0.8; // 확대,축소
float pre_scope = 0.0;
float* z = (float*)malloc(width * width * sizeof(float));
unsigned char Image[width * width * 3];
float dx = 0.0, dy = 0.0; // mouse move
bool TimeDir = true, Mode = true; // 확대 축소 방향 . time/scroll 
int key_s = 0; // 방정식 바꾸는 key
int key_d = 0; // 차수 늘리는 key
float Zoom = -50.0;
int StartPt[2];

// 콜백 함수
void Keyboard(unsigned char key, int x, int y);
void Mouse(int button, int state, int x, int y);
void MouseMove(int x, int y);
void MouseWheel(int button, int dir, int x, int y);
void Reshape(int w, int h);
void Timer(int id);


// 연산자 오버라이딩
////////////////////////////////////////////////////////////////////////////////////////

float abs(ucomplex x) {
    return sqrt(x.real * x.real + x.imag * x.imag);
}

ucomplex operator*(ucomplex x, ucomplex y) {
    ucomplex c = {
    (x.real * y.real - x.imag * y.imag),
    (x.imag * y.real + x.real * y.imag)
    };
    return c;
}

ucomplex operator*(ucomplex x, float dy) {
    ucomplex y = { dy, 0.0 };
    return x * y;
}

ucomplex operator/(ucomplex x, ucomplex y) {
    ucomplex c = {
    (x.real * y.real + x.imag * y.imag) / (y.real * y.real + y.imag * y.imag),
    (x.imag * y.real - x.real * y.imag) / (y.real * y.real + y.imag * y.imag)
    };
    return c;
}

ucomplex operator+(ucomplex x, ucomplex y) {
    ucomplex c = {
    (x.real + y.real),
    (x.imag + y.imag)
    };
    return c;
}

ucomplex operator+(ucomplex x, float dy) {
    ucomplex y = { dy, 0.0 };
    return x + y;
}

ucomplex operator-(ucomplex x, ucomplex y) {
    ucomplex c = {
    (x.real - y.real),
    (x.imag - y.imag)
    };
    return c;
}

ucomplex operator-(ucomplex x, float dy) {
    ucomplex y = { dy, 0.0 };
    return x - y;
}
/////////////////////////////////////////////////////////////////////////////////////////

ucomplex f(ucomplex x)
{
    // 1. 초기 방정식: x^3 - 1
    ucomplex d = x * x * x - 1.0;
    if (key_s == 0)
    {
        ucomplex temp_d = x * x * x;
        if (key_d > 0) {
            for (int i = 0; i < key_d; i++)
                temp_d = temp_d * x;
        }
        d = temp_d - 1.0;
    }
    // 2. x^3 - 2x^2 + 2
    else if (key_s == 1)
    {
        d = x * x * x - x * x * 2.0 + 2.0;
    }
    // 3. x^8 + 15x^4 - 16
    else if (key_s == 2)
    {
        d = x * x * x * x * x * x * x * x + x * x * x * x * 15.0 - 16.0;
    }
    // 4. sin(x)
    else if (key_s == 3)
    {
        typedef std::complex<float> dcomplex;
        dcomplex a(x.real, x.imag);
        d = {
           sin(a).real(),
           sin(a).imag()
        };
    }
    return d;
}

ucomplex df(ucomplex x) // f 미분값
{
    ucomplex d = x * x * 3;
    // 1. 3x^2
    if (key_s == 0)
    {
        ucomplex temp_d = x * x;
        float de = 3.0;
        if (key_d > 0) {
            for (int i = 0; i < key_d; i++) {
                temp_d = temp_d * x;
                de = de + 1.0;
            }
        }
        d = temp_d * de;

    }
    // 2. 3x^2 - 4x
    else if (key_s == 1)
    {
        d = x * x * 3 - x * 4.0;
    }
    // 3. 8x^7 +60x^3
    else if (key_s == 2)
    {
        d = x * x * x * x * x * x * x * 8.0 + x * x * x * 60.0;
    }
    // 4. cos(x)
    else if (key_s == 3)
    {
        typedef std::complex<float> dcomplex;
        dcomplex a(x.real, x.imag);
        d = {
           cos(a).real(),
           cos(a).imag()
        };
    }
    return d;
}

//사용자 정의함수
__int64 GetMicroSecond();
float newton(ucomplex x0, float eps, int maxiter);
void MathCPU();
void Display();

int main(int argc, char** argv) {

    // opengl 초기화
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB);
    //glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);

    // 윈도우 설정
    glutInitWindowSize(width, width);
    //glutInitWindowSize(700, 700);
    glutInitWindowPosition(0, 0);
    glutCreateWindow("Newton");
    glClearColor(1.0, 1.0, 1.0, 1.0);

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

    glutMainLoop();
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
    if (key == 'e')
    {
        if (key_d > 0)
            key_d--;
    }
    // 모드 바꾸기
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
    dx += -(x - StartPt[0]) * 0.01; // x 이동값
    dy += -(StartPt[1] - y) * 0.01; // y 이동값
}

void MouseWheel(int button, int dir, int x, int y)
{
    // Mode = false 일때
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

float newton(ucomplex x0, float eps, int maxiter) {
    ucomplex x = x0;
    int iter = 0;
    // 픽셀의 고유 iter값 계산
    while (abs(f(x)) > eps&& iter <= maxiter) {
        iter++;
        x = x - f(x) / df(x); // 점화식
    }
    return iter;
}

void Timer(int id)
{
    // Mode = true 일때 시간에따라 확대,축소
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

void MathCPU() {

    // 범위 설정
    float xmin = -2, xmax = 2;
    float ymin = -2, ymax = 2;

    // 각 픽셀간의 간격을 설정 
    int xsteps = width, ysteps = width;
    float hx = (xmax - xmin) / xsteps, hy = (ymax - ymin) / ysteps;
    float eps = 0.0001; // eps보다 작아지면 더이상 진행하지 않음
    int maxiter = 255; // 최대 iter값

    float scopenum = 15 * sin(scope); // 확대 축소 범위: -15 ~ 15
    float x, y;
    y = ymin;
    // 픽셀별 뉴턴값 계산
    for (int i = 0; i < ysteps; i++) {
        x = xmin;
        for (int j = 0; j < xsteps; j++) {
            ucomplex  xy = { x * scopenum + dx,y * scopenum + dy }; // 복소수값 할당
            z[i * width + j] = newton(xy, eps, maxiter); // 뉴턴값 계산
            x += hx; // + 1열
        }
        y += hy; // +1행
    }
}

// width * width 픽셀에 색상을 지정
void Display() {
    glClearColor(1, 1, 1, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    __int64 _st = GetMicroSecond();
    MathCPU();

    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            float color = (z[i * width + j] / (float)256);
            int offset = (i * width + j) * 3;
            Image[offset] = fmod(color * 13, 1.0f) * 255;
            Image[offset + 1] = fmod(color * 33, 1.0f) * 255;
            Image[offset + 2] = fmod(color * 49, 1.0f) * 255;
        }
    }
    glDrawPixels(width, width, GL_RGB, GL_UNSIGNED_BYTE, Image);

    printf("Elapsed time: %I64d mico sec \n", GetMicroSecond() - _st);

    glFinish();
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
/*
//width * width 점을 찍어서 렌더링
void Display() {
   glClear(GL_COLOR_BUFFER_BIT);
   glBegin(GL_POINTS);

   __int64 _st = GetMicroSecond();
   MathCPU();

   float xmin = -2, xmax = 2;
   float ymin = -2, ymax = 2;

   int xsteps = width, ysteps = width;
   float hx = (xmax - xmin) / xsteps, hy = (ymax - ymin) / ysteps;

   float x, y;
   float max = z[0];
   float min = z[0];

   for (int i = 0; i < width; i++) {
     for (int j = 0; j < width; j++) {
       if (z[i * width + j] >= max) max = z[i * width + j];
       if (z[i * width + j] <= min) min = z[i * width + j];
     }
   }

   y = ymin;
   for (int i = 0; i < width; i++) {
     x = xmin;
      for (int j = 0; j < width; j++) {
         float color = (z[i * width + j]) / (256);
         glColor3d(fmod(color * 13, 1.0f),
            fmod(color * 33, 1.0f),
            fmod(color * 49, 1.0f));
         glVertex2d(x, y);
       x += hx;
      }
     y += hy;
   }

   printf("Elapsed time: %I64d mico sec \n", GetMicroSecond() - _st);
   glEnd();
   glutSwapBuffers();
}
*/