% --- Пункт 1: Підготовка даних ---
% Визначаємо межі та крок 0.5
x1_range = -3:0.5:3;
x2_range = -2:0.5:2;
[X1, X2] = meshgrid(x1_range, x2_range);

% Обчислюємо значення функції f(x1, x2)
% f = 2*exp(2 - 5*x1^2) + 5*(x1 - x2^2)
F_values = 2*exp(2 - 5*X1.^2) + 5*(X1 - X2.^2);

% Формуємо вхідні (X) та цільові (T) масиви
X_train = [X1(:)'; X2(:)']; 
T_train = F_values(:)';

% --- Пункт 2: Дослідження залежності кількості нейронів ---
errors = [0.1, 0.01, 0.001];
for goal_err = errors
    % Створюємо мережу. Параметр spread візьмемо 1.0 за замовчуванням
    net = newrb(X_train, T_train, goal_err);
    
    % Виводимо кількість нейронів у першому шарі
    num_neurons = net.layers{1}.size;
    fprintf('Для помилки %.3f створено нейронів: %d\n', goal_err, num_neurons);
end

% Частина 2
% Використовуємо останню навчену мережу (net), яка має 97 нейронів

% --- Пункт 3: Тестові дані з кроком 0.25 (в 2 рази щільніше) ---
x1_test_range = -3:0.25:3;
x2_test_range = -2:0.25:2;
[X1_t, X2_test] = meshgrid(x1_test_range, x2_test_range);
X_test = [X1_t(:)'; X2_test(:)'];

% --- Пункт 4: Обчислення виходу мережі на тестових даних ---
Y_test = net(X_test);

% --- Пункт 5: Обчислення еталона за формулою та помилки ---
% Використовуємо ту саму формулу для нових точок
T_test = 2*exp(2 - 5*X1_t(:)'.^2) + 5*(X1_t(:)' - X2_test(:)'.^2);
error_vec = T_test - Y_test;

% --- Пункт 6: Побудова 3D-графіка апроксимованої функції ---
figure;
subplot(1,2,1);
Y_surf = reshape(Y_test, size(X1_t));
surf(X1_t, X2_test, Y_surf);
title('Апроксимована функція (RBF)');
xlabel('x1'); ylabel('x2'); zlabel('f(x1,x2)');

% --- Пункт 7: Графік помилки апроксимації ---
subplot(1,2,2);
E_surf = reshape(error_vec, size(X1_t));
surf(X1_t, X2_test, E_surf);
title('Графік помилки апроксимації');
xlabel('x1'); ylabel('x2'); zlabel('Error');