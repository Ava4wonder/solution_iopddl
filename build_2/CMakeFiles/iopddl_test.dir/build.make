# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.31

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/ava/Downloads/iopddl

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/ava/Downloads/iopddl/build_2

# Include any dependencies generated for this target.
include CMakeFiles/iopddl_test.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/iopddl_test.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/iopddl_test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/iopddl_test.dir/flags.make

CMakeFiles/iopddl_test.dir/codegen:
.PHONY : CMakeFiles/iopddl_test.dir/codegen

CMakeFiles/iopddl_test.dir/iopddl_test.cc.o: CMakeFiles/iopddl_test.dir/flags.make
CMakeFiles/iopddl_test.dir/iopddl_test.cc.o: /Users/ava/Downloads/iopddl/iopddl_test.cc
CMakeFiles/iopddl_test.dir/iopddl_test.cc.o: CMakeFiles/iopddl_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/ava/Downloads/iopddl/build_2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/iopddl_test.dir/iopddl_test.cc.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/iopddl_test.dir/iopddl_test.cc.o -MF CMakeFiles/iopddl_test.dir/iopddl_test.cc.o.d -o CMakeFiles/iopddl_test.dir/iopddl_test.cc.o -c /Users/ava/Downloads/iopddl/iopddl_test.cc

CMakeFiles/iopddl_test.dir/iopddl_test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/iopddl_test.dir/iopddl_test.cc.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/ava/Downloads/iopddl/iopddl_test.cc > CMakeFiles/iopddl_test.dir/iopddl_test.cc.i

CMakeFiles/iopddl_test.dir/iopddl_test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/iopddl_test.dir/iopddl_test.cc.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/ava/Downloads/iopddl/iopddl_test.cc -o CMakeFiles/iopddl_test.dir/iopddl_test.cc.s

CMakeFiles/iopddl_test.dir/iopddl.cc.o: CMakeFiles/iopddl_test.dir/flags.make
CMakeFiles/iopddl_test.dir/iopddl.cc.o: /Users/ava/Downloads/iopddl/iopddl.cc
CMakeFiles/iopddl_test.dir/iopddl.cc.o: CMakeFiles/iopddl_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/ava/Downloads/iopddl/build_2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/iopddl_test.dir/iopddl.cc.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/iopddl_test.dir/iopddl.cc.o -MF CMakeFiles/iopddl_test.dir/iopddl.cc.o.d -o CMakeFiles/iopddl_test.dir/iopddl.cc.o -c /Users/ava/Downloads/iopddl/iopddl.cc

CMakeFiles/iopddl_test.dir/iopddl.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/iopddl_test.dir/iopddl.cc.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/ava/Downloads/iopddl/iopddl.cc > CMakeFiles/iopddl_test.dir/iopddl.cc.i

CMakeFiles/iopddl_test.dir/iopddl.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/iopddl_test.dir/iopddl.cc.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/ava/Downloads/iopddl/iopddl.cc -o CMakeFiles/iopddl_test.dir/iopddl.cc.s

CMakeFiles/iopddl_test.dir/solver.cc.o: CMakeFiles/iopddl_test.dir/flags.make
CMakeFiles/iopddl_test.dir/solver.cc.o: /Users/ava/Downloads/iopddl/solver.cc
CMakeFiles/iopddl_test.dir/solver.cc.o: CMakeFiles/iopddl_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/ava/Downloads/iopddl/build_2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/iopddl_test.dir/solver.cc.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/iopddl_test.dir/solver.cc.o -MF CMakeFiles/iopddl_test.dir/solver.cc.o.d -o CMakeFiles/iopddl_test.dir/solver.cc.o -c /Users/ava/Downloads/iopddl/solver.cc

CMakeFiles/iopddl_test.dir/solver.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/iopddl_test.dir/solver.cc.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/ava/Downloads/iopddl/solver.cc > CMakeFiles/iopddl_test.dir/solver.cc.i

CMakeFiles/iopddl_test.dir/solver.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/iopddl_test.dir/solver.cc.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/ava/Downloads/iopddl/solver.cc -o CMakeFiles/iopddl_test.dir/solver.cc.s

CMakeFiles/iopddl_test.dir/submission_v0_scso_solver.cc.o: CMakeFiles/iopddl_test.dir/flags.make
CMakeFiles/iopddl_test.dir/submission_v0_scso_solver.cc.o: /Users/ava/Downloads/iopddl/submission_v0_scso_solver.cc
CMakeFiles/iopddl_test.dir/submission_v0_scso_solver.cc.o: CMakeFiles/iopddl_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/ava/Downloads/iopddl/build_2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/iopddl_test.dir/submission_v0_scso_solver.cc.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/iopddl_test.dir/submission_v0_scso_solver.cc.o -MF CMakeFiles/iopddl_test.dir/submission_v0_scso_solver.cc.o.d -o CMakeFiles/iopddl_test.dir/submission_v0_scso_solver.cc.o -c /Users/ava/Downloads/iopddl/submission_v0_scso_solver.cc

CMakeFiles/iopddl_test.dir/submission_v0_scso_solver.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/iopddl_test.dir/submission_v0_scso_solver.cc.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/ava/Downloads/iopddl/submission_v0_scso_solver.cc > CMakeFiles/iopddl_test.dir/submission_v0_scso_solver.cc.i

CMakeFiles/iopddl_test.dir/submission_v0_scso_solver.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/iopddl_test.dir/submission_v0_scso_solver.cc.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/ava/Downloads/iopddl/submission_v0_scso_solver.cc -o CMakeFiles/iopddl_test.dir/submission_v0_scso_solver.cc.s

# Object files for target iopddl_test
iopddl_test_OBJECTS = \
"CMakeFiles/iopddl_test.dir/iopddl_test.cc.o" \
"CMakeFiles/iopddl_test.dir/iopddl.cc.o" \
"CMakeFiles/iopddl_test.dir/solver.cc.o" \
"CMakeFiles/iopddl_test.dir/submission_v0_scso_solver.cc.o"

# External object files for target iopddl_test
iopddl_test_EXTERNAL_OBJECTS =

iopddl_test: CMakeFiles/iopddl_test.dir/iopddl_test.cc.o
iopddl_test: CMakeFiles/iopddl_test.dir/iopddl.cc.o
iopddl_test: CMakeFiles/iopddl_test.dir/solver.cc.o
iopddl_test: CMakeFiles/iopddl_test.dir/submission_v0_scso_solver.cc.o
iopddl_test: CMakeFiles/iopddl_test.dir/build.make
iopddl_test: lib/libgmock_main.a
iopddl_test: lib/libgtest_main.a
iopddl_test: external/abseil-cpp/absl/status/libabsl_statusor.a
iopddl_test: lib/libgmock.a
iopddl_test: lib/libgtest.a
iopddl_test: external/abseil-cpp/absl/status/libabsl_status.a
iopddl_test: external/abseil-cpp/absl/strings/libabsl_cord.a
iopddl_test: external/abseil-cpp/absl/strings/libabsl_cordz_info.a
iopddl_test: external/abseil-cpp/absl/strings/libabsl_cord_internal.a
iopddl_test: external/abseil-cpp/absl/strings/libabsl_cordz_functions.a
iopddl_test: external/abseil-cpp/absl/profiling/libabsl_exponential_biased.a
iopddl_test: external/abseil-cpp/absl/strings/libabsl_cordz_handle.a
iopddl_test: external/abseil-cpp/absl/synchronization/libabsl_synchronization.a
iopddl_test: external/abseil-cpp/absl/synchronization/libabsl_graphcycles_internal.a
iopddl_test: external/abseil-cpp/absl/synchronization/libabsl_kernel_timeout_internal.a
iopddl_test: external/abseil-cpp/absl/base/libabsl_tracing_internal.a
iopddl_test: external/abseil-cpp/absl/time/libabsl_time.a
iopddl_test: external/abseil-cpp/absl/time/libabsl_civil_time.a
iopddl_test: external/abseil-cpp/absl/time/libabsl_time_zone.a
iopddl_test: external/abseil-cpp/absl/crc/libabsl_crc_cord_state.a
iopddl_test: external/abseil-cpp/absl/crc/libabsl_crc32c.a
iopddl_test: external/abseil-cpp/absl/crc/libabsl_crc_internal.a
iopddl_test: external/abseil-cpp/absl/crc/libabsl_crc_cpu_detect.a
iopddl_test: external/abseil-cpp/absl/debugging/libabsl_leak_check.a
iopddl_test: external/abseil-cpp/absl/types/libabsl_bad_optional_access.a
iopddl_test: external/abseil-cpp/absl/debugging/libabsl_stacktrace.a
iopddl_test: external/abseil-cpp/absl/base/libabsl_strerror.a
iopddl_test: external/abseil-cpp/absl/debugging/libabsl_symbolize.a
iopddl_test: external/abseil-cpp/absl/debugging/libabsl_debugging_internal.a
iopddl_test: external/abseil-cpp/absl/debugging/libabsl_demangle_internal.a
iopddl_test: external/abseil-cpp/absl/debugging/libabsl_demangle_rust.a
iopddl_test: external/abseil-cpp/absl/debugging/libabsl_decode_rust_punycode.a
iopddl_test: external/abseil-cpp/absl/debugging/libabsl_utf8_for_code_point.a
iopddl_test: external/abseil-cpp/absl/base/libabsl_malloc_internal.a
iopddl_test: external/abseil-cpp/absl/strings/libabsl_str_format_internal.a
iopddl_test: external/abseil-cpp/absl/strings/libabsl_strings.a
iopddl_test: external/abseil-cpp/absl/strings/libabsl_strings_internal.a
iopddl_test: external/abseil-cpp/absl/strings/libabsl_string_view.a
iopddl_test: external/abseil-cpp/absl/base/libabsl_base.a
iopddl_test: external/abseil-cpp/absl/base/libabsl_spinlock_wait.a
iopddl_test: external/abseil-cpp/absl/numeric/libabsl_int128.a
iopddl_test: external/abseil-cpp/absl/base/libabsl_throw_delegate.a
iopddl_test: external/abseil-cpp/absl/types/libabsl_bad_variant_access.a
iopddl_test: external/abseil-cpp/absl/base/libabsl_raw_logging_internal.a
iopddl_test: external/abseil-cpp/absl/base/libabsl_log_severity.a
iopddl_test: CMakeFiles/iopddl_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/ava/Downloads/iopddl/build_2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable iopddl_test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/iopddl_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/iopddl_test.dir/build: iopddl_test
.PHONY : CMakeFiles/iopddl_test.dir/build

CMakeFiles/iopddl_test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/iopddl_test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/iopddl_test.dir/clean

CMakeFiles/iopddl_test.dir/depend:
	cd /Users/ava/Downloads/iopddl/build_2 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/ava/Downloads/iopddl /Users/ava/Downloads/iopddl /Users/ava/Downloads/iopddl/build_2 /Users/ava/Downloads/iopddl/build_2 /Users/ava/Downloads/iopddl/build_2/CMakeFiles/iopddl_test.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/iopddl_test.dir/depend

