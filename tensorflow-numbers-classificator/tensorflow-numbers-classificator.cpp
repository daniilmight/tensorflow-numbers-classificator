#include <tensorflow/c/c_api.h>  // C API TensorFlow
#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <string>

// ------------------------------------------------------------
// Структура для хранения цифры и её вероятности
// ------------------------------------------------------------
struct DigitProb {
    int digit;     // предсказанная цифра
    float prob;    // вероятность (score) этой цифры
};



// ------------------------------------------------------------
// Проверка статуса выполнения операций TensorFlow
// Если произошла ошибка - выводим её и завершаем программу
// ------------------------------------------------------------
void check_status(TF_Status* status) {
    if (TF_GetCode(status) != TF_OK) {
        std::cerr << "TensorFlow error: " << TF_Message(status) << std::endl;
        std::exit(1);
    }
}

// ------------------------------------------------------------
// Чтение бинарного файла с входными данными
// sample_input.bin хранит одно изображение MNIST (28x28, float32)
// ------------------------------------------------------------
std::vector<float> read_input_bin(const std::string& path) {
    const size_t sz = 28 * 28 * 1;  // размер изображения (MNIST, 28x28x1)
    std::vector<float> buf(sz);

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open " << path << std::endl;
        std::exit(1);
    }

    in.read(reinterpret_cast<char*>(buf.data()), sz * sizeof(float));
    if (!in) {
        std::cerr << "Failed to read bytes" << std::endl;
        std::exit(1);
    }
    return buf;
}

// ------------------------------------------------------------
// Загрузка графа TensorFlow из .pb файла
// ------------------------------------------------------------
TF_Graph* load_graph(const std::string& pb_path) {
    TF_Graph* graph = TF_NewGraph();
    TF_Status* status = TF_NewStatus();

    // Загружаем .pb файл в память
    std::ifstream f(pb_path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::cerr << "Cannot open pb file: " << pb_path << std::endl;
        std::exit(1);
    }
    std::streamsize size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!f.read(buffer.data(), size)) {
        std::cerr << "Failed reading pb file\n";
        std::exit(1);
    }

    // Импортируем граф в объект TF_Graph
    TF_Buffer* pb_buf = TF_NewBufferFromString(buffer.data(), size);
    TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();
    TF_GraphImportGraphDef(graph, pb_buf, opts, status);
    check_status(status);

    // Освобождение временных ресурсов
    TF_DeleteImportGraphDefOptions(opts);
    TF_DeleteBuffer(pb_buf);
    TF_DeleteStatus(status);

    return graph;
}

// ------------------------------------------------------------
// Главная программа
// ------------------------------------------------------------
int main() {

    std::string python_script = "python preprocess_image.py";
    std::string input_image = "my_digit.png";
    std::string output_bin = "export/sample_input.bin";

    std::string command = python_script + " " + input_image + " " + output_bin;  // формируем команду
    int ret = std::system(command.c_str());  // запускаем Python скрипт
    if (ret != 0) {
        std::cerr << "Python preprocessing failed!" << std::endl;
        return 1;
    }

    // Пути к данным
    std::string export_dir = "D:/Codding/tensorflow-numbers-classificator/tensorflow-numbers-classificator/export";
    std::string pb_path = export_dir + "/mnist_frozen_graph.pb";
    std::string sample_file = export_dir + "/sample_input.bin";

    std::cout << "Loading graph from: " << pb_path << std::endl;
    TF_Graph* graph = load_graph(pb_path);


    // Создание сессии для выполнения инференса
    TF_Status* status = TF_NewStatus();
    TF_SessionOptions* sess_opts = TF_NewSessionOptions();
    TF_Session* session = TF_NewSession(graph, sess_opts, status);
    check_status(status);

    // Чтение входного изображения
    std::vector<float> input_data = read_input_bin(sample_file);

    // Создание входного тензора: размер [1,28,28,1], тип float32
    int64_t dims[4] = { 1,28,28,1 };
    TF_Tensor* input_tensor = TF_AllocateTensor(
        TF_FLOAT, dims, 4, input_data.size() * sizeof(float));
    std::memcpy(TF_TensorData(input_tensor), input_data.data(), input_data.size() * sizeof(float));

    // Поиск операций по имени
    // --- Находим вход и выход по именам из графа ---
    TF_Operation* input_op = TF_GraphOperationByName(graph, "x");
    TF_Operation* output_op = TF_GraphOperationByName(graph, "sequential_1/predictions_1/Softmax");

    if (!input_op || !output_op) {
        std::cerr << "Не удалось найти input или output операции!" << std::endl;
        return 1;
    }

    // Настройка входов и выходов
    TF_Output inputs[1] = { {input_op,0} };
    TF_Tensor* input_vals[1] = { input_tensor };
    TF_Output outputs[1] = { {output_op,0} };
    TF_Tensor* output_vals[1] = { nullptr };

    // Запуск инференса
    TF_SessionRun(session, nullptr,
        inputs, input_vals, 1,    // входы
        outputs, output_vals, 1,  // выходы
        nullptr, 0, nullptr,      // нет операций для исполнения кроме инференса
        status);
    check_status(status);

    // Получение предсказаний
    float* out_data = static_cast<float*>(TF_TensorData(output_vals[0]));

    // Вывод всех вероятностей (для каждой цифры от 0 до 9)
    std::cout << "All predictions:\n";
    std::vector<DigitProb> probs;
    for (int i = 0; i < 10; i++) {
        std::cout << "Digit " << i << ": " << out_data[i] << "\n";
        probs.push_back({ i, out_data[i] });
    }

    // Сортировка по убыванию вероятности
    std::sort(probs.begin(), probs.end(), [](const DigitProb& a, const DigitProb& b) {
        return a.prob > b.prob;
        });

    // Вывод топ-3 предсказаний
    std::cout << "\nTop 3 predictions:\n";
    for (int i = 0; i < 3; i++) {
        std::cout << probs[i].digit << " (score=" << probs[i].prob << ")\n";
    }

    // Финальный вывод - самая вероятная цифра
    std::cout << "\nPredicted digit: "
        << probs[0].digit << " (score=" << probs[0].prob << ")\n";

    // Очистка ресурсов
    TF_DeleteTensor(input_tensor);
    TF_DeleteTensor(output_vals[0]);
    TF_CloseSession(session, status);
    TF_DeleteSession(session, status);
    TF_DeleteGraph(graph);
    TF_DeleteSessionOptions(sess_opts);
    TF_DeleteStatus(status);

    return 0;
}
