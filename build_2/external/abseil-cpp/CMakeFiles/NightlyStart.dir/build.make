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

# Utility rule file for NightlyStart.

# Include any custom commands dependencies for this target.
include external/abseil-cpp/CMakeFiles/NightlyStart.dir/compiler_depend.make

# Include the progress variables for this target.
include external/abseil-cpp/CMakeFiles/NightlyStart.dir/progress.make

external/abseil-cpp/CMakeFiles/NightlyStart:
	cd /Users/ava/Downloads/iopddl/build_2/external/abseil-cpp && /opt/homebrew/bin/ctest -D NightlyStart

external/abseil-cpp/CMakeFiles/NightlyStart.dir/codegen:
.PHONY : external/abseil-cpp/CMakeFiles/NightlyStart.dir/codegen

NightlyStart: external/abseil-cpp/CMakeFiles/NightlyStart
NightlyStart: external/abseil-cpp/CMakeFiles/NightlyStart.dir/build.make
.PHONY : NightlyStart

# Rule to build all files generated by this target.
external/abseil-cpp/CMakeFiles/NightlyStart.dir/build: NightlyStart
.PHONY : external/abseil-cpp/CMakeFiles/NightlyStart.dir/build

external/abseil-cpp/CMakeFiles/NightlyStart.dir/clean:
	cd /Users/ava/Downloads/iopddl/build_2/external/abseil-cpp && $(CMAKE_COMMAND) -P CMakeFiles/NightlyStart.dir/cmake_clean.cmake
.PHONY : external/abseil-cpp/CMakeFiles/NightlyStart.dir/clean

external/abseil-cpp/CMakeFiles/NightlyStart.dir/depend:
	cd /Users/ava/Downloads/iopddl/build_2 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/ava/Downloads/iopddl /Users/ava/Downloads/iopddl/external/abseil-cpp /Users/ava/Downloads/iopddl/build_2 /Users/ava/Downloads/iopddl/build_2/external/abseil-cpp /Users/ava/Downloads/iopddl/build_2/external/abseil-cpp/CMakeFiles/NightlyStart.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : external/abseil-cpp/CMakeFiles/NightlyStart.dir/depend

