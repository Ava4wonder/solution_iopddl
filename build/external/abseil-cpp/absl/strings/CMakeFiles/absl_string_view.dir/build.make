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
include external/abseil-cpp/absl/strings/CMakeFiles/absl_string_view.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include external/abseil-cpp/absl/strings/CMakeFiles/absl_string_view.dir/compiler_depend.make

# Include the progress variables for this target.
include external/abseil-cpp/absl/strings/CMakeFiles/absl_string_view.dir/progress.make

# Include the compile flags for this target's objects.
include external/abseil-cpp/absl/strings/CMakeFiles/absl_string_view.dir/flags.make

external/abseil-cpp/absl/strings/CMakeFiles/absl_string_view.dir/codegen:
.PHONY : external/abseil-cpp/absl/strings/CMakeFiles/absl_string_view.dir/codegen

external/abseil-cpp/absl/strings/CMakeFiles/absl_string_view.dir/string_view.cc.o: external/abseil-cpp/absl/strings/CMakeFiles/absl_string_view.dir/flags.make
external/abseil-cpp/absl/strings/CMakeFiles/absl_string_view.dir/string_view.cc.o: /Users/ava/Downloads/iopddl/external/abseil-cpp/absl/strings/string_view.cc
external/abseil-cpp/absl/strings/CMakeFiles/absl_string_view.dir/string_view.cc.o: external/abseil-cpp/absl/strings/CMakeFiles/absl_string_view.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/ava/Downloads/iopddl/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object external/abseil-cpp/absl/strings/CMakeFiles/absl_string_view.dir/string_view.cc.o"
	cd /Users/ava/Downloads/iopddl/build/external/abseil-cpp/absl/strings && /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT external/abseil-cpp/absl/strings/CMakeFiles/absl_string_view.dir/string_view.cc.o -MF CMakeFiles/absl_string_view.dir/string_view.cc.o.d -o CMakeFiles/absl_string_view.dir/string_view.cc.o -c /Users/ava/Downloads/iopddl/external/abseil-cpp/absl/strings/string_view.cc

external/abseil-cpp/absl/strings/CMakeFiles/absl_string_view.dir/string_view.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/absl_string_view.dir/string_view.cc.i"
	cd /Users/ava/Downloads/iopddl/build/external/abseil-cpp/absl/strings && /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/ava/Downloads/iopddl/external/abseil-cpp/absl/strings/string_view.cc > CMakeFiles/absl_string_view.dir/string_view.cc.i

external/abseil-cpp/absl/strings/CMakeFiles/absl_string_view.dir/string_view.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/absl_string_view.dir/string_view.cc.s"
	cd /Users/ava/Downloads/iopddl/build/external/abseil-cpp/absl/strings && /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/ava/Downloads/iopddl/external/abseil-cpp/absl/strings/string_view.cc -o CMakeFiles/absl_string_view.dir/string_view.cc.s

# Object files for target absl_string_view
absl_string_view_OBJECTS = \
"CMakeFiles/absl_string_view.dir/string_view.cc.o"

# External object files for target absl_string_view
absl_string_view_EXTERNAL_OBJECTS =

external/abseil-cpp/absl/strings/libabsl_string_view.a: external/abseil-cpp/absl/strings/CMakeFiles/absl_string_view.dir/string_view.cc.o
external/abseil-cpp/absl/strings/libabsl_string_view.a: external/abseil-cpp/absl/strings/CMakeFiles/absl_string_view.dir/build.make
external/abseil-cpp/absl/strings/libabsl_string_view.a: external/abseil-cpp/absl/strings/CMakeFiles/absl_string_view.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/ava/Downloads/iopddl/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libabsl_string_view.a"
	cd /Users/ava/Downloads/iopddl/build/external/abseil-cpp/absl/strings && $(CMAKE_COMMAND) -P CMakeFiles/absl_string_view.dir/cmake_clean_target.cmake
	cd /Users/ava/Downloads/iopddl/build/external/abseil-cpp/absl/strings && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/absl_string_view.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
external/abseil-cpp/absl/strings/CMakeFiles/absl_string_view.dir/build: external/abseil-cpp/absl/strings/libabsl_string_view.a
.PHONY : external/abseil-cpp/absl/strings/CMakeFiles/absl_string_view.dir/build

external/abseil-cpp/absl/strings/CMakeFiles/absl_string_view.dir/clean:
	cd /Users/ava/Downloads/iopddl/build/external/abseil-cpp/absl/strings && $(CMAKE_COMMAND) -P CMakeFiles/absl_string_view.dir/cmake_clean.cmake
.PHONY : external/abseil-cpp/absl/strings/CMakeFiles/absl_string_view.dir/clean

external/abseil-cpp/absl/strings/CMakeFiles/absl_string_view.dir/depend:
	cd /Users/ava/Downloads/iopddl/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/ava/Downloads/iopddl /Users/ava/Downloads/iopddl/external/abseil-cpp/absl/strings /Users/ava/Downloads/iopddl/build /Users/ava/Downloads/iopddl/build/external/abseil-cpp/absl/strings /Users/ava/Downloads/iopddl/build/external/abseil-cpp/absl/strings/CMakeFiles/absl_string_view.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : external/abseil-cpp/absl/strings/CMakeFiles/absl_string_view.dir/depend

