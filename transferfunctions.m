b1 = 100;
b2 = 1000;
b3 = 800;
m1 = 0.1;
m2 = 0.5;
m3 = 0.05;

F = 10;
Fenv = 5;
Ff = 3;

k = 3e5;

tf1 = tf(1 ,[m1, b1, k]); 
dampratio1 = b1/(2*sqrt(k*m1));

tf2 = tf(1, [m2, b2, k]); 
dampratio2 = b2/(2*sqrt(k*m2));

tf3 = tf(1, [m3, b3, k]); 
dampratio3 = b3/(2*sqrt(k*m3));

A = [0 1 0 0 0 0; -k/m1 -b1/m1 0 0 k/m1 0; 0 0 0 1 0 0; 0 0 -k/m2 -b2/m2 k/m2 0; 0 0 0 0 0 1; k/m3 0 k/m3 0 -2*k/m3 -b3/m3];
B = [0; 1; 0; 0; 0; 0];
C = eye(6);
D = 0;

sys = ss(A,B,C,D);
tf_all = ss2tf(sys.A, sys.B, sys.C, sys.D);
