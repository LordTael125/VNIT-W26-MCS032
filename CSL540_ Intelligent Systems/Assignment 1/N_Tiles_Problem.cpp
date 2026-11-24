#include <algorithm>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace std;

int countInversions(const vector<int> &Matrix) {
  int inversions = 0;
  int total = Matrix.size();
  for (int i = 0; i < total; i++) {
    if (Matrix[i] == 0)
      continue;
    for (int j = i + 1; j < total; j++) {
      if (Matrix[j] == 0)
        continue;
      if (Matrix[i] > Matrix[j]) {
        inversions++;
      }
    }
  }
  return inversions;
}

bool isSolvable(const vector<int> &Matrix, int n) {
  int inversions = countInversions(Matrix);

  if (n % 2 != 0) {
    return (inversions % 2 == 0);
  } else {
    int Blank_idx = 0;
    while (Matrix[Blank_idx] != 0)
      Blank_idx++;
    int zero_row = Blank_idx / n;
    return ((inversions + zero_row) % 2 == 0);
  }
}

vector<int> getGoalState(int n) {
  vector<int> goal(n * n);
  for (int i = 0; i < n * n; i++) {
    goal[i] = i;
  }
  return goal;
}

vector<int> getRandomBoard(int n, const vector<int> &goal) {
  vector<int> board = goal;
  random_device rd;
  mt19937 g(rd());

  do {
    shuffle(board.begin(), board.end(), g);
  } while (!isSolvable(board, n) || board == goal);

  return board;
}

bool dls(vector<int> &Matrix, int n, int limit, int prev_blank,
         const vector<int> &goal, vector<string> &path,
         vector<vector<int>> &states) {
  if (Matrix == goal)
    return true;
  if (limit <= 0)
    return false;

  int Blank_idx = 0;
  while (Matrix[Blank_idx] != 0)
    Blank_idx++;

  // cout<<Blank_idx<<" "<<endl;
  int row = Blank_idx / n;
  int col = Blank_idx % n;

  int dr[] = {-1, 1, 0, 0};
  int dc[] = {0, 0, -1, 1};

  string tile_actions[] = {"DOWN", "UP", "RIGHT", "LEFT"};

  for (int i = 0; i < 4; i++) {
    int nr = row + dr[i];
    int nc = col + dc[i];

    if (nr >= 0 && nr < n && nc >= 0 && nc < n) {
      int new_idx = nr * n + nc;

      if (new_idx == prev_blank)
        continue;

      int moved_tile = Matrix[new_idx];
      string move_description =
          "Tile " + to_string(moved_tile) + " moves " + tile_actions[i];

      swap(Matrix[Blank_idx], Matrix[new_idx]);
      path.push_back(move_description);
      states.push_back(Matrix);

      if (dls(Matrix, n, limit - 1, Blank_idx, goal, path, states)) {
        return true;
      }

      states.pop_back();
      path.pop_back();
      swap(Matrix[Blank_idx], Matrix[new_idx]);
    }
  }
  return false;
}

void printBoard(const vector<int> &Matrix, int n) {
  for (int i = 0; i < n * n; i++) {
    if (Matrix[i] == 0)
      cout << "_ ";
    else
      cout << Matrix[i] << " ";
    if ((i + 1) % n == 0)
      cout << "\n";
  }
  cout << "\n";
}

int main() {
  int n;
  cout << "Enter the size of the board N (e.g., 2, 3, 4): ";
  if (!(cin >> n) || n < 2) {
    cout << "Invalid board size. N must be at least 2.\n";
    return 1;
  }

  vector<int> goall = getGoalState(n);
  vector<int> Matrix = getRandomBoard(n, goall);

  cout << "\n Goal State:\n";
  printBoard(goall, n);

  cout << " Initial Solvable Board \n";
  printBoard(Matrix, n);

  cout << "Searching for solution using Iterative Deepening Search (IDS)\n";
  vector<string> path;
  vector<vector<int>> states;
  bool solved = false;

  int max_depth = (n == 2) ? 6 : (n == 3 ? 31 : 80);

  for (int limit = 0; limit <= max_depth; limit++) {
    path.clear();
    states.clear();
    vector<int> current = Matrix;
    if (dls(current, n, limit, -1, goall, path, states)) {
      solved = true;
      break;
    }
  }

  if (solved) {
    cout << "\nSolved in " << path.size() << " moves!\n";
    cout << "==============================\n";

    cout << "Step 0 (Initial State):\n";
    printBoard(Matrix, n);

    for (size_t i = 0; i < path.size(); i++) {
      cout << "Step " << (i + 1) << " [" << path[i] << "]:\n";
      printBoard(states[i], n);
    }
  } else {
    cout << "No solution found within the depth limit (" << max_depth << ").\n";
  }

  return 0;
}