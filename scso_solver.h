// scso_solver.h
#ifndef SCSO_SOLVER_H
#define SCSO_SOLVER_H

#include "iopddl.h"
#include "absl/time/time.h"

namespace iopddl {

// RunSCSO implements the new solution logic (using a population-based optimization)
// and returns a solution (a vector of strategy indices).
// Solution RunSCSO(const Problem &problem);
Solution RunSCSO(const Problem &problem, absl::Time start_time, absl::Duration timeout);
// Solution RunSCSO(const Problem &problem, absl::Duration timeout = absl::InfiniteDuration());

}  // namespace iopddl

#endif  // SCSO_SOLVER_H
