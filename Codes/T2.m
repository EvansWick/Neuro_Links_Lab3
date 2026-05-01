% --- Крок 1: Підготовка даних ---
prprob_my;


% --- Крок 2: Створення та навчання мережі ---
net = feedforwardnet(31, 'trainlm'); % 31 нейрон у прихованому шарі
net.trainParam.goal = 1e-6;          % Цільова помилка
net.divideFcn = '';                  % Використовувати всі дані
net = train(net, alphabet, targets); % Навчання на "чистих" даних

% --- Крок 3: Дослідження стійкості до шуму (п. 6-7) ---
noise_levels = 0:0.05:0.5;           % Рівні шуму від 0 до 0.5
mean_errors = zeros(size(noise_levels)); % Масив для збереження помилок

for i = 1:length(noise_levels)
    std_dev = noise_levels(i);       % Поточне відхилення шуму
    total_error = 0;
    
    for char_idx = 1:26              % Для кожної з 26 літер
        for trial = 1:10             % По 10 зашумлених зразків
            % Додавання шуму за допомогою randn
            noisy_input = alphabet(:, char_idx) + std_dev * randn(35, 1);
            
            % Отримання виходу мережі
            y_out = net(noisy_input);
            
            % Розрахунок помилки як евклідової норми різниці
            total_error = total_error + norm(targets(:, char_idx) - y_out);
        end
    end
    % Обчислення середньої сумарної помилки для рівня шуму
    mean_errors(i) = total_error / (26 * 10); 
end

% --- Крок 4: Візуалізація результатів ---
figure;
plot(noise_levels, mean_errors, 'r-s', 'LineWidth', 2);
title('Залежність помилки розпізнавання від рівня шуму');
xlabel('Інтенсивність шуму (Standard Deviation)');
ylabel('Середня сумарна помилка (Euclidean Norm)');
grid on;

% Tests
fake_letter = rand(35, 1); % Повністю випадковий шум
prediction = net(fake_letter);
[val, idx] = max(prediction); 
fprintf('Мережа впевнена на %.2f%%, що це літера №%d\n', val*100, idx);
figure;
subplot(1,2,1); plotchar(fake_letter); title('Літера з випадковим шумом');
subplot(1,2,2); plotchar(alphabet(:, idx)); title('Літера з яку визначила мережа');


% Test 2
std_dev = 0.15;
test_char = alphabet(:, 10); % Візьмемо літеру E (5-та в алфавіті)
noisy_E = test_char + std_dev * randn(35, 1);

% Перевіримо прогноз
out = net(noisy_E);
[~, predicted_idx] = max(out);
disp(['Мережа розпізнала це як літеру №: ', num2str(predicted_idx)]);

% Візуалізуємо, що бачить мережа
figure;
subplot(1,3,1); plotchar(test_char); title('Оригінал J');
subplot(1,3,2); plotchar(noisy_E); title('Що бачить мережа (шум 0.15)');
subplot(1,3,3); plotchar(alphabet(:, predicted_idx)); title('Що розпізнала мережа ');

