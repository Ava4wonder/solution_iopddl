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
CMAKE_BINARY_DIR = /Users/ava/Downloads/iopddl/build

# Include any dependencies generated for this target.
include external/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_functions.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include external/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_functions.dir/compiler_depend.make

# Include the progress variables for this target.
include external/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_functions.dir/progress.make

# Include the compile flags for this target's objects.
include external/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_functions.dir/flags.make

external/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_functions.dir/codegen:
.PHONY : external/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_functions.dir/codegen

external/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_functions.dir/internal/cordz_functions.cc.o: external/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_functions.dir/flags.make
external/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_functions.dir/internal/cordz_functions.cc.o: /Users/ava/Downloads/iopddl/external/abseil-cpp/absl/strings/internal/cordz_functions.cc
external/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_functions.dir/internal/cordz_functions.cc.o: external/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_functions.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/ava/Downloads/iopddl/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object external/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_functions.dir/internal/cordz_functions.cc.o"
	cd /Users/ava/Downloads/iopddl/build/external/abseil-cpp/absl/strings && /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT external/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_functions.dir/internal/cordz_functions.cc.o -MF CMakeFiles/absl_cordz_functions.dir/internal/cordz_functions.cc.o.d -o CMakeFiles/absl_cordz_functions.dir/internal/cordz_functions.cc.o -c /Users/ava/Downloads/iopddl/external/abseil-cpp/absl/strings/internal/cordz_functions.cc

external/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_functions.dir/internal/cordz_functions.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/absl_cordz_functions.dir/internal/cordz_functions.cc.i"
	cd /Users/ava/Downloads/iopddl/build/external/abseil-cpp/absl/strings && /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/ava/Downloads/iopddl/external/abseil-cpp/absl/strings/internal/cordz_functions.cc > CMakeFiles/absl_cordz_functions.dir/internal/cordz_functions.cc.i

external/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_functions.dir/internal/cordz_functions.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/absl_cordz_functions.dir/internal/cordz_functions.cc.s"
	cd /Users/ava/Downloads/iopddl/build/external/abseil-cpp/absl/strings && /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/ava/Downloads/iopddl/external/abseil-cpp/absl/strings/internal/cordz_functions.cc -o CMakeFiles/absl_cordz_functions.dir/internal/cordz_functions.cc.s

# Object files for target absl_cordz_functions
absl_cordz_functions_OBJECTS = \
"CMakeFiles/absl_cordz_functions.dir/internal/cordz_functions.cc.o"

# External object files for target absl_cordz_functions
absl_cordz_functions_EXTERNAL_OBJECTS =

external/abseil-cpp/absl/strings/libabsl_cordz_functions.a: external/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_functions.dir/internal/cordz_functions.cc.o
external/abseil-cpp/absl/strings/libabsl_cordz_functions.a: external/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_functions.dir/build.make
external/abseil-cpp/absl/strings/libabsl_cordz_functions.a: external/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_functions.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/ava/Downloads/iopddl/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libabsl_cordz_functions.a"
	cd /Users/ava/Downloads/iopddl/build/external/abseil-cpp/absl/strings && $(CMAKE_COMMAND) -P CMakeFiles/absl_cordz_functions.dir/cmake_clean_target.cmake
	cd /Users/ava/Downloads/iopddl/build/external/abseil-cpp/absl/strings && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/absl_cordz_functions.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
external/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_functions.dir/build: external/abseil-cpp/absl/strings/libabsl_cordz_functions.a
.PHONY : external/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_functions.dir/build

external/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_functions.dir/clean:
	cd /Users/ava/Downloads/iopddl/build/external/abseil-cpp/absl/strings && $(CMAKE_COMMAND) -P CMakeFiles/absl_cordz_functions.dir/cmake_clean.cmake
.PHONY : external/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_functions.dir/clean

external/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_functions.dir/depend:
	cd /Users/ava/Downloads/iopddl/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/ava/Downloads/iopddl /Users/ava/Downloads/iopddl/external/abseil-cpp/absl/strings /Users/ava/Downloads/iopddl/build /Users/ava/Downloads/iopddl/build/external/abseil-cpp/absl/strings /Users/ava/Downloads/iopddl/build/external/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_functions.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : external/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_functions.dir/depend

