% Define values 

num_elements = 3;
radius = 0.4e-3; 
density = 26.0; 
b = 100; 
E = 1.5e11; 
length = 1; 
area = pi*radius^2; 
volume = area*length;
mass = volume*density/num_elements; 
k = E*area/length;
T_in = 10;

M = mass*eye(num_elements); 
B = 2*b*eye(num_elements);
B(1, 1) = b; 
D = zeros(num_elements, num_elements+1); 
K = zeros(num_elements+1, num_elements);

for i=1:num_elements-1
   
    B(i, i+1) = -b;
    B(i+1, i) = -b;
    
end

for i=1:num_elements 
   
    D(i,i) = 1;
    D(i,i+1) = -1;
    
end

for i=1:num_elements 
   
    K(i,i) = -k;
    K(i+1, i) = k;
    
end

K(1, 1) = 0;
% K(num_elements+1, num_elements) = 10;
% B(num_elements, num_elements) = 0;
% M(num_elements, num_elements) = 0.5;

% Need to change mass up over here. 

% Issue in this script is with the total displacement created, but system is already stable. Need to check Python
% script.

A_p = [zeros(num_elements) eye(num_elements)];
A_v = [inv(M)*D*K -inv(M)*B];

A_ss = [A_p; A_v];
B_ss = zeros(num_elements*2, 1);
B_ss(num_elements + 1) = (1/mass)*T_in;
C_ss = eye(num_elements*2);
D_ss = 0;

sys = ss(A_ss,B_ss,C_ss,D_ss);
tf_all = ss2tf(sys.A, sys.B, sys.C, sys.D);
