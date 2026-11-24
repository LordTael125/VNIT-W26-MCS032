#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

// ================= MODEL DEFINITION =================
struct MLPModel {
    int input_size = 0;
    int hidden_size = 0;
    int output_size = 1;

    // Layer 1 (Input -> Hidden) [hidden_size][input_size]
    vector<vector<double>> W1;
    vector<double> B1;

    // Layer 2 (Hidden -> Output) [hidden_size]
    vector<double> W2;
    double B2 = 0.0;
};

// ================= ACTIVATION FUNCTIONS =================
inline double sigmoid(double z) {
    if (z > 20.0) return 1.0;
    if (z < -20.0) return 0.0;
    return 1.0 / (1.0 + exp(-z));
}

inline double sigmoid_derivative(double a) {
    return a * (1.0 - a);
}

// ================= UNIVERSAL ROBUST CSV READER =================
bool read_csv(const string &filepath, vector<vector<double>> &X, vector<int> &y, vector<string> &feature_names) {
    X.clear();
    y.clear();
    feature_names.clear();

    ifstream file(filepath);
    if (!file.is_open()) {
        cerr << "\n[CSV Reader Error] Could not open file: \"" << filepath << "\"\n";
        cerr << "  -> Please verify the file path and ensure it exists.\n";
        return false;
    }

    string line;
    int line_number = 0;
    bool is_first_line = true;
    size_t expected_cols = 0;

    while (getline(file, line)) {
        line_number++;
        if (line.empty()) continue;
        if (line.back() == '\r') line.pop_back();
        if (line.empty()) continue;

        char delim = (line.find(';') != string::npos && line.find(',') == string::npos) ? ';' : ',';

        stringstream ss(line);
        string token;
        vector<string> tokens;

        while (getline(ss, token, delim)) {
            size_t start = token.find_first_not_of(" \t");
            size_t end = token.find_last_not_of(" \t");
            if (start != string::npos && end != string::npos) {
                tokens.push_back(token.substr(start, end - start + 1));
            } else {
                tokens.push_back("");
            }
        }

        if (tokens.empty()) continue;

        if (is_first_line) {
            is_first_line = false;
            expected_cols = tokens.size();

            bool is_header = false;
            try {
                (void)stod(tokens[0]);
            } catch (...) {
                is_header = true;
            }

            if (is_header) {
                for (size_t i = 0; i < tokens.size() - 1; ++i) {
                    feature_names.push_back(tokens[i]);
                }
                continue;
            } else {
                for (size_t i = 0; i < tokens.size() - 1; ++i) {
                    feature_names.push_back("x" + to_string(i + 1));
                }
            }
        }

        if (tokens.size() != expected_cols) {
            cerr << "[Warning] Line " << line_number << " has " << tokens.size() 
                 << " columns (expected " << expected_cols << "). Skipping.\n";
            continue;
        }

        try {
            vector<double> row_features;
            for (size_t i = 0; i < tokens.size() - 1; ++i) {
                row_features.push_back(stod(tokens[i]));
            }
            int target = static_cast<int>(stod(tokens.back()));
            X.push_back(row_features);
            y.push_back(target);
        } catch (const exception &e) {
            cerr << "[Warning] Line " << line_number << " contains non-numeric values. Skipping.\n";
        }
    }

    file.close();

    if (X.empty()) {
        cerr << "[CSV Reader Error] No valid data rows found in file: \"" << filepath << "\"\n";
        return false;
    }

    return true;
}

bool read_csv(const string &filepath, vector<vector<double>> &X, vector<int> &y) {
    vector<string> unused;
    return read_csv(filepath, X, y, unused);
}

// ================= MODULE 1: TRAINING =================
MLPModel train(const string &train_file, int hidden_nodes = 4, double learning_rate = 0.25, int max_epochs = 250) {
    cout << "\n===================================================\n";
    cout << " [MODULE 1: TRAINING MLP (BACKPROPAGATION)]\n";
    cout << " User-Provided Training File: " << train_file << "\n";
    cout << " Hidden Neurons: " << hidden_nodes << " | Learning Rate: " << learning_rate 
         << " | Max Epochs: " << max_epochs << "\n";
    cout << "===================================================\n";

    vector<vector<double>> X;
    vector<int> y;
    vector<string> feature_names;

    if (!read_csv(train_file, X, y, feature_names)) {
        cerr << "[Fatal Error] Training aborted due to file reading error.\n";
        return MLPModel();
    }

    size_t num_samples = X.size();
    size_t num_features = X[0].size();

    cout << ">> Dataset Loaded: " << num_samples << " rows, " << num_features << " features.\n";
    cout << "   Feature Names: ";
    for (size_t i = 0; i < feature_names.size(); ++i) {
        cout << feature_names[i] << (i + 1 < feature_names.size() ? ", " : "\n");
    }

    MLPModel model;
    model.input_size = static_cast<int>(num_features);
    model.hidden_size = hidden_nodes;
    model.output_size = 1;

    // Weight Initialization with deterministic seed
    mt19937 rng(42);
    double limit1 = 1.0 / sqrt(model.input_size);
    uniform_real_distribution<double> dist1(-limit1, limit1);

    model.W1.assign(model.hidden_size, vector<double>(model.input_size));
    model.B1.assign(model.hidden_size, 0.0);
    for (int h = 0; h < model.hidden_size; ++h) {
        for (int j = 0; j < model.input_size; ++j) {
            model.W1[h][j] = dist1(rng);
        }
        model.B1[h] = dist1(rng) * 0.1;
    }

    double limit2 = 1.0 / sqrt(model.hidden_size);
    uniform_real_distribution<double> dist2(-limit2, limit2);
    model.W2.assign(model.hidden_size, 0.0);
    for (int h = 0; h < model.hidden_size; ++h) {
        model.W2[h] = dist2(rng);
    }
    model.B2 = dist2(rng) * 0.1;

    ofstream metrics_file("mlp_metrics.csv");
    metrics_file << "Epoch,E_average\n";

    cout << left << setw(8) << "\nEpoch" 
         << setw(20) << "E_average (MSE)" 
         << setw(18) << "Accuracy (%)" << "\n";
    cout << string(46, '-') << "\n";

    for (int epoch = 1; epoch <= max_epochs; ++epoch) {
        double total_squared_error = 0.0;
        int correct_predictions = 0;

        for (size_t i = 0; i < num_samples; ++i) {
            // Forward Pass
            vector<double> hidden_activations(model.hidden_size, 0.0);
            for (int h = 0; h < model.hidden_size; ++h) {
                double z_h = model.B1[h];
                for (int j = 0; j < model.input_size; ++j) {
                    z_h += model.W1[h][j] * X[i][j];
                }
                hidden_activations[h] = sigmoid(z_h);
            }

            double z_out = model.B2;
            for (int h = 0; h < model.hidden_size; ++h) {
                z_out += model.W2[h] * hidden_activations[h];
            }
            double y_hat = sigmoid(z_out);

            double error = static_cast<double>(y[i]) - y_hat;
            total_squared_error += 0.5 * (error * error);

            int predicted_class = (y_hat >= 0.5) ? 1 : 0;
            if (predicted_class == y[i]) correct_predictions++;

            // Backpropagation
            double delta_out = error * sigmoid_derivative(y_hat);

            vector<double> delta_hidden(model.hidden_size, 0.0);
            for (int h = 0; h < model.hidden_size; ++h) {
                delta_hidden[h] = (delta_out * model.W2[h]) * sigmoid_derivative(hidden_activations[h]);
            }

            // Update Layer 2
            for (int h = 0; h < model.hidden_size; ++h) {
                model.W2[h] += learning_rate * delta_out * hidden_activations[h];
            }
            model.B2 += learning_rate * delta_out;

            // Update Layer 1
            for (int h = 0; h < model.hidden_size; ++h) {
                for (int j = 0; j < model.input_size; ++j) {
                    model.W1[h][j] += learning_rate * delta_hidden[h] * X[i][j];
                }
                model.B1[h] += learning_rate * delta_hidden[h];
            }
        }

        double e_avg = total_squared_error / num_samples;
        double train_acc = (static_cast<double>(correct_predictions) / num_samples) * 100.0;
        metrics_file << epoch << "," << fixed << setprecision(6) << e_avg << "\n";

        if (epoch % 25 == 0 || epoch == 1 || epoch == max_epochs) {
            cout << left << setw(8) << epoch 
                 << setw(20) << fixed << setprecision(6) << e_avg 
                 << setw(18) << fixed << setprecision(2) << train_acc << "\n";
        }
    }
    metrics_file.close();

    cout << "\n[Training Complete - MLP Parameters Saved Successfully]\n";

    // Auto-generate Python visualization script
    ofstream py_file("plot_mlp.py");
    py_file << "import sys\n"
            << "try:\n"
            << "    import pandas as pd\n"
            << "    import matplotlib.pyplot as plt\n"
            << "    df = pd.read_csv('mlp_metrics.csv')\n"
            << "    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))\n"
            << "    ax1.axis('tight'); ax1.axis('off')\n"
            << "    table_data = df.tail(15).round(5)\n"
            << "    table = ax1.table(cellText=table_data.values, colLabels=table_data.columns, cellLoc='center', loc='center')\n"
            << "    table.scale(1, 1.4)\n"
            << "    ax1.set_title('MLP Backpropagation: E_average', fontweight='bold')\n"
            << "    ax2.plot(df['Epoch'], df['E_average'], color='#6a1b9a', linewidth=2.5, label='Backprop MSE Loss')\n"
            << "    ax2.set_xlabel('Epoch', fontsize=12)\n"
            << "    ax2.set_ylabel('E_average (Mean Squared Error)', fontsize=12)\n"
            << "    ax2.set_title('MLP: Backpropagation Loss Reduction Curve', fontweight='bold')\n"
            << "    ax2.grid(True, linestyle='--', alpha=0.6)\n"
            << "    ax2.legend()\n"
            << "    plt.tight_layout()\n"
            << "    plt.savefig('mlp_convergence_plot.png', dpi=150)\n"
            << "    print('[Python] Saved plot to mlp_convergence_plot.png')\n"
            << "    plt.close()\n"
            << "except ImportError:\n"
            << "    print('[Notice] Matplotlib/pandas not installed. Run: pip3 install pandas matplotlib')\n";
    py_file.close();

    cout << "\n[Executing Generated Python Visualization Script...]\n";
    system("python3 plot_mlp.py");

    return model;
}

// ================= MODULE 2: TESTING =================
void test(const MLPModel &model, const vector<string> &test_files) {
    cout << "\n===================================================\n";
    cout << " [MODULE 2: TESTING MLP (BACKPROPAGATION)]\n";
    cout << " Total Test Files Provided: " << test_files.size() << "\n";
    cout << " Decision Rule: y_hat >= 0.5 -> Class 1, else Class 0\n";
    cout << "===================================================\n";

    for (size_t f = 0; f < test_files.size(); ++f) {
        const string &filename = test_files[f];
        cout << "\n>>> Evaluating User Test File " << (f + 1) << ": " << filename << " <<<\n";

        vector<vector<double>> X;
        vector<int> y;
        vector<string> feature_names;

        if (!read_csv(filename, X, y, feature_names)) {
            cerr << "[Warning] Skipping unreadable test file: " << filename << "\n";
            continue;
        }

        if (static_cast<int>(X[0].size()) != model.input_size) {
            cerr << "[Error] Feature count mismatch in " << filename << "!\n";
            continue;
        }

        int correct_count = 0;
        cout << left << setw(8) << "Sample" 
             << setw(32) << "Features [Inputs]" 
             << setw(16) << "Output Prob" 
             << setw(12) << "Predicted" 
             << setw(12) << "Actual" 
             << setw(10) << "Result" << "\n";
        cout << string(90, '-') << "\n";

        for (size_t i = 0; i < X.size(); ++i) {
            vector<double> hidden_activations(model.hidden_size, 0.0);
            for (int h = 0; h < model.hidden_size; ++h) {
                double z_h = model.B1[h];
                for (int j = 0; j < model.input_size; ++j) {
                    z_h += model.W1[h][j] * X[i][j];
                }
                hidden_activations[h] = sigmoid(z_h);
            }

            double z_out = model.B2;
            for (int h = 0; h < model.hidden_size; ++h) {
                z_out += model.W2[h] * hidden_activations[h];
            }
            double y_hat = sigmoid(z_out);

            int pred = (y_hat >= 0.5) ? 1 : 0;
            bool is_correct = (pred == y[i]);
            if (is_correct) correct_count++;

            stringstream ss_in;
            ss_in << "[";
            for (size_t j = 0; j < X[i].size(); ++j) {
                ss_in << fixed << setprecision(2) << X[i][j] << (j + 1 < X[i].size() ? ", " : "");
            }
            ss_in << "]";

            cout << left << setw(8) << (i + 1)
                 << setw(32) << ss_in.str()
                 << setw(16) << fixed << setprecision(4) << y_hat
                 << setw(12) << pred
                 << setw(12) << y[i]
                 << setw(10) << (is_correct ? "CORRECT" : "WRONG") << "\n";
        }

        double accuracy = (static_cast<double>(correct_count) / X.size()) * 100.0;
        cout << string(90, '-') << "\n";
        cout << "Test Summary for [" << filename << "]: "
             << correct_count << "/" << X.size() << " Correct | Accuracy: "
             << fixed << setprecision(2) << accuracy << "%\n";
    }
}

// ================= MAIN FUNCTION =================
int main(int argc, char* argv[]) {
    string train_file;
    vector<string> test_files;

    if (argc >= 3) {
        train_file = argv[1];
        for (int i = 2; i < argc; ++i) {
            test_files.push_back(argv[i]);
        }
        cout << "[Info] Using user-provided CSV files from command line:\n";
        cout << "  Train file: " << train_file << "\n";
        cout << "  Test files: (" << test_files.size() << " files)\n";
    } else if (argc == 2) {
        train_file = argv[1];
        test_files.push_back(argv[1]);
        cout << "[Info] Using user-provided CSV file: " << train_file << "\n";
    } else {
        cout << "===============================================================\n";
        cout << " NOTICE: You can provide ANY custom CSV file via command line:\n";
        cout << "   Usage: ./mlp <path_to_train.csv> <test1.csv> [test2.csv ...]\n";
        cout << " Example: ./mlp my_train.csv my_test1.csv my_test2.csv\n";
        cout << "===============================================================\n";
        cout << "[Info] No command line arguments detected. Loading default assignment files...\n";
        train_file = "mlp_train.csv";
        test_files = {"mlp_test1.csv", "mlp_test2.csv"};
    }

    MLPModel model = train(train_file, 4, 0.25, 250);

    if (!test_files.empty() && model.input_size > 0) {
        test(model, test_files);
    }

    return 0;
}
