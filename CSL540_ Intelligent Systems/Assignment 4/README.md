# Assignment IV: DPLL Algorithm for Propositional Inference
**Course**: CSL540 - Intelligent Systems  
**Submission Email**: `uad.ai.assignment@gmail.com`  
**Subject**: `<Your Name>, <Your Enrollment Number>, Assignment 4`

---

## 1. Overview & Propositional Inference via Refutation

DPLL (Davis-Putnam-Logemann-Loveland) is a complete backtracking algorithm for deciding propositional satisfiability (SAT).

### Checking Entailment ($KB \models \alpha$) using DPLL:
By the **Refutation Principle** (Proof by Contradiction):
$$KB \models \alpha \iff (KB \land \neg \alpha) \text{ is UNSATISFIABLE}$$

1. We convert the Knowledge Base $KB$ into Conjunctive Normal Form (CNF clauses).
2. We negate the query $\alpha$ ($\neg \alpha$).
   - If $\alpha$ is a literal $L$, $\neg \alpha = \neg L$ is added as a unit clause.
   - If $\alpha$ is a clause $(L_1 \lor L_2 \dots)$, $\neg \alpha = (\neg L_1 \land \neg L_2 \dots)$ is added as separate unit clauses.
3. We run DPLL on the augmented clause set $KB \cup \{\neg \alpha\}$:
   - **UNSATISFIABLE**: $KB \models \alpha$ **HOLDS (TRUE)**. No model exists where $KB$ is true and $\alpha$ is false.
   - **SATISFIABLE**: $KB \not\models \alpha$ **FAILS (FALSE)**. The satisfying model is an explicit **counterexample** where $KB$ is satisfied but $\alpha$ is false.

---

## 2. DPLL Core Components Implemented

1. **Early Termination**:
   - Returns `true` if all clauses are satisfied under the current partial model.
   - Returns `false` (backtracks) if any clause is conflicting (all literals false).
2. **Pure Symbol Elimination**:
   - Identifies symbols that appear with only one polarity across all currently unresolved clauses and assigns them that polarity.
3. **Unit Clause Propagation**:
   - Identifies clauses with only one unassigned literal where all others are false, forcing that remaining literal to true.
4. **Splitting / Backtracking Search**:
   - Recursively branches on unassigned symbols.

---

## 3. Compilation & Execution

### Compilation:
```bash
g++ -std=c++17 -O2 dpll_inference.cpp -o dpll_inference
```

### Run Demonstration Suite (Directly from Terminal):
```bash
./dpll_inference --demo
```

### Run Interactive Menu:
```bash
./dpll_inference
```
The menu allows you to:
1. Run the demonstration suite (Modus Ponens, Fallacy check, Wumpus World deduction, SAT/UNSAT formulas).
2. Interactively input a custom Knowledge Base (KB) and query $\alpha$ to check $KB \models \alpha$.
3. Interactively solve any general CNF SAT problem and inspect the truth assignment model.
