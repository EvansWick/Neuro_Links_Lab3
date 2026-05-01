% --- Пункт 8: Створення PNN для класифікації літер ---
prprob_my; % Виклик твого файлу з alphabet та targets

% Створюємо імовірнісну мережу. Spread за замовчуванням 0.1
net_pnn = newpnn(alphabet, targets); 

% --- Пункт 9: Дослідження стійкості PNN до шуму ---
noise_lvls = 0:0.05:0.5;
pnn_mean_errors = zeros(size(noise_lvls));

for i = 1:length(noise_lvls)
    sd = noise_lvls(i);
    total_e = 0;
    for c_idx = 1:26
        for t_trial = 1:10
            % Генеруємо зашумлений вхід
            n_in = alphabet(:, c_idx) + sd * randn(35, 1);
            
            % Отримуємо відповідь PNN
            y_pnn = net_pnn(n_in);
            
            % Рахуємо евклідову норму помилки
            total_e = total_e + norm(targets(:, c_idx) - y_pnn);
        end
    end
    pnn_mean_errors(i) = total_e / (26 * 10);
end

% Візуалізація стійкості PNN
figure;
plot(noise_lvls, pnn_mean_errors, 'b-o', 'LineWidth', 2);
title('Стійкість імовірнісної мережі (PNN) до шуму');
xlabel('Рівень шуму (Standard Deviation)');
ylabel('Середня помилка (Euclidean Norm)');
grid on;