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
include external/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_info.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include external/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_info.dir/compiler_depend.make

# Include the progress variables for this target.
include external/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_info.dir/progress.make

# Include the compile flags for this target's objects.
include external/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_info.dir/flags.make

external/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_info.dir/codegen:
.PHONY : external/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_info.dir/codegen

external/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_info.dir/internal/cordz_info.cc.o: external/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_info.dir/flags.make
external/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_info.dir/internal/cordz_info.cc.o: /Users/ava/Downloads/iopddl/external/abseil-cpp/absl/strings/internal/cordz_info.cc
external/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_info.dir/internal/cordz_info.cc.o: external/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_info.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/ava/Downloads/iopddl/build_2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object external/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_info.dir/internal/cordz_info.cc.o"
	cd /Users/ava/Downloads/iopddl/build_2/external/abseil-cpp/absl/strings && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT external/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_info.dir/internal/cordz_info.cc.o -MF CMakeFiles/absl_cordz_info.dir/internal/cordz_info.cc.o.d -o CMakeFiles/absl_cordz_info.dir/internal/cordz_info.cc.o -c /Users/ava/Downloads/iopddl/external/abseil-cpp/absl/strings/internal/cordz_info.cc

external/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_info.dir/internal/cordz_info.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/absl_cordz_info.dir/internal/cordz_info.cc.i"
	cd /Users/ava/Downloads/iopddl/build_2/external/abseil-cpp/absl/strings && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/ava/Downloads/iopddl/external/abseil-cpp/absl/strings/internal/cordz_info.cc > CMakeFiles/absl_cordz_info.dir/internal/cordz_info.cc.i

external/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_info.dir/internal/cordz_info.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/absl_cordz_info.dir/internal/cordz_info.cc.s"
	cd /Users/ava/Downloads/iopddl/build_2/external/abseil-cpp/absl/strings && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/ava/Downloads/iopddl/external/abseil-cpp/absl/strings/internal/cordz_info.cc -o CMakeFiles/absl_cordz_info.dir/internal/cordz_info.cc.s

# Object files for target absl_cordz_info
absl_cordz_info_OBJECTS = \
"CMakeFiles/absl_cordz_info.dir/internal/cordz_info.cc.o"

# External object files for target absl_cordz_info
absl_cordz_info_EXTERNAL_OBJECTS =

external/abseil-cpp/absl/strings/libabsl_cordz_info.a: external/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_info.dir/internal/cordz_info.cc.o
external/abseil-cpp/absl/strings/libabsl_cordz_info.a: external/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_info.dir/build.make
external/abseil-cpp/absl/strings/libabsl_cordz_info.a: external/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_info.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/ava/Downloads/iopddl/build_2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libabsl_cordz_info.a"
	cd /Users/ava/Downloads/iopddl/build_2/external/abseil-cpp/absl/strings && $(CMAKE_COMMAND) -P CMakeFiles/absl_cordz_info.dir/cmake_clean_target.cmake
	cd /Users/ava/Downloads/iopddl/build_2/external/abseil-cpp/absl/strings && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/absl_cordz_info.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
external/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_info.dir/build: external/abseil-cpp/absl/strings/libabsl_cordz_info.a
.PHONY : external/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_info.dir/build

external/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_info.dir/clean:
	cd /Users/ava/Downloads/iopddl/build_2/external/abseil-cpp/absl/strings && $(CMAKE_COMMAND) -P CMakeFiles/absl_cordz_info.dir/cmake_clean.cmake
.PHONY : external/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_info.dir/clean

external/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_info.dir/depend:
	cd /Users/ava/Downloads/iopddl/build_2 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/ava/Downloads/iopddl /Users/ava/Downloads/iopddl/external/abseil-cpp/absl/strings /Users/ava/Downloads/iopddl/build_2 /Users/ava/Downloads/iopddl/build_2/external/abseil-cpp/absl/strings /Users/ava/Downloads/iopddl/build_2/external/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_info.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : external/abseil-cpp/absl/strings/CMakeFiles/absl_cordz_info.dir/depend

