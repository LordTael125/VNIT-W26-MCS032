#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <map>
#include <set>

using namespace std;

struct DPLLSolver {
    map<string, int> name_to_id;
    vector<string> id_to_name;

    int parse_literal(string tok) {
        bool neg = (!tok.empty() && (tok[0] == '~' || tok[0] == '!' || tok[0] == '-'));
        if (neg) tok = tok.substr(1);
        if (!name_to_id.count(tok)) {
            id_to_name.push_back(tok);
            name_to_id[tok] = id_to_name.size();
        }
        return neg ? -name_to_id[tok] : name_to_id[tok];
    }

    string lit_str(int lit) const {
        return (lit < 0 ? "~" : "") + id_to_name[abs(lit) - 1];
    }

    string clause_str(const vector<int>& c) const {
        string s = "(";
        for (size_t i = 0; i < c.size(); ++i)
            s += lit_str(c[i]) + (i + 1 < c.size() ? " | " : "");
        return s + ")";
    }

    // +1: SATISFIED, -1: CONFLICT, 0: UNRESOLVED
    int eval_clause(const vector<int>& c, const map<int, bool>& m) const {
        int unassigned = 0;
        for (int lit : c) {
            auto it = m.find(abs(lit));
            if (it != m.end()) {
                if (it->second == (lit > 0)) return 1;
            } else {
                unassigned++;
            }
        }
        return unassigned == 0 ? -1 : 0;
    }

    bool find_pure(const vector<vector<int>>& clauses, const vector<int>& syms,
                   const map<int, bool>& m, int& var, bool& val) const {
        for (int s : syms) {
            int pos = 0, neg = 0;
            for (const auto& c : clauses) {
                if (eval_clause(c, m) == 1) continue;
                for (int lit : c)
                    if (abs(lit) == s) (lit > 0 ? pos : neg)++;
            }
            if (pos > 0 && neg == 0) { var = s; val = true; return true; }
            if (neg > 0 && pos == 0) { var = s; val = false; return true; }
        }
        return false;
    }

    bool find_unit(const vector<vector<int>>& clauses, const map<int, bool>& m,
                   int& var, bool& val, int& idx) const {
        for (size_t i = 0; i < clauses.size(); ++i) {
            if (eval_clause(clauses[i], m) != 0) continue;
            int cnt = 0, last_lit = 0;
            for (int lit : clauses[i])
                if (!m.count(abs(lit))) { cnt++; last_lit = lit; }
            if (cnt == 1) {
                var = abs(last_lit); val = (last_lit > 0); idx = i;
                return true;
            }
        }
        return false;
    }

    bool dpll(const vector<vector<int>>& clauses, vector<int> syms,
              map<int, bool>& m, vector<string>& trace) const {
        bool all_sat = true;
        for (const auto& c : clauses) {
            int st = eval_clause(c, m);
            if (st == -1) {
                trace.push_back("Conflict: " + clause_str(c) + " is FALSE");
                return false;
            }
            if (st == 0) all_sat = false;
        }
        if (all_sat) {
            trace.push_back("All clauses satisfied.");
            return true;
        }

        // 1. Pure Symbol Elimination
        int var, idx; bool val;
        if (find_pure(clauses, syms, m, var, val)) {
            trace.push_back("Pure Symbol: " + lit_str(var) + " = " + (val ? "TRUE" : "FALSE"));
            vector<int> rem;
            for (int s : syms) if (s != var) rem.push_back(s);
            map<int, bool> next = m; next[var] = val;
            if (dpll(clauses, rem, next, trace)) { m = next; return true; }
            return false;
        }

        // 2. Unit Clause Propagation
        if (find_unit(clauses, m, var, val, idx)) {
            trace.push_back("Unit Prop " + clause_str(clauses[idx]) + ": " + lit_str(var) + " = " + (val ? "TRUE" : "FALSE"));
            vector<int> rem;
            for (int s : syms) if (s != var) rem.push_back(s);
            map<int, bool> next = m; next[var] = val;
            if (dpll(clauses, rem, next, trace)) { m = next; return true; }
            return false;
        }

        // 3. Branching / Backtracking
        if (syms.empty()) return false;
        int p = syms[0];
        vector<int> rest(syms.begin() + 1, syms.end());

        trace.push_back("Branch: " + lit_str(p) + " = TRUE");
        map<int, bool> m_true = m; m_true[p] = true;
        if (dpll(clauses, rest, m_true, trace)) { m = m_true; return true; }

        trace.push_back("Backtrack: " + lit_str(p) + " = FALSE");
        map<int, bool> m_false = m; m_false[p] = false;
        if (dpll(clauses, rest, m_false, trace)) { m = m_false; return true; }

        return false;
    }

    // KB |= alpha  <==>  (KB ^ ~alpha) is UNSATISFIABLE
    bool check_entailment(vector<vector<int>> kb, const vector<int>& query,
                          map<int, bool>& model, vector<string>& trace) const {
        for (int lit : query) kb.push_back({-lit});
        set<int> sym_set;
        for (const auto& c : kb) for (int lit : c) sym_set.insert(abs(lit));
        return !dpll(kb, vector<int>(sym_set.begin(), sym_set.end()), model, trace);
    }
};

int main() {
    DPLLSolver solver;
    vector<vector<int>> kb;
    string line, tok;

    cout << "Enter KB clauses (space-separated, e.g. '~P Q'). Type 'done' to finish:\n";
    while (getline(cin, line)) {
        if (line == "done" || (line.empty() && !kb.empty())) break;
        stringstream ss(line);
        vector<int> clause;
        while (ss >> tok) clause.push_back(solver.parse_literal(tok));
        if (!clause.empty()) kb.push_back(clause);
    }

    cout << "Enter Query (alpha): ";
    getline(cin, line);
    stringstream ss_q(line);
    vector<int> query;
    while (ss_q >> tok) query.push_back(solver.parse_literal(tok));

    map<int, bool> model;
    vector<string> trace;
    bool entailed = solver.check_entailment(kb, query, model, trace);

    cout << "\n--- DPLL Steps on (KB ^ ~alpha) ---\n";
    for (size_t i = 0; i < trace.size(); ++i)
        cout << i + 1 << ". " << trace[i] << "\n";

    cout << "\n--- Result ---\n";
    if (entailed) {
        cout << "UNSATISFIABLE -> KB |= alpha HOLDS (TRUE)\n";
    } else {
        cout << "SATISFIABLE -> KB |= alpha DOES NOT HOLD (FALSE)\nCounterexample:\n";
        for (size_t i = 0; i < solver.id_to_name.size(); ++i) {
            int var = i + 1;
            cout << "  " << solver.id_to_name[i] << " = "
                 << (model.count(var) ? (model[var] ? "TRUE" : "FALSE") : "ANY") << "\n";
        }
    }
    return 0;
}