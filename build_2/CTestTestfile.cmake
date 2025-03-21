# CMake generated Testfile for 
# Source directory: /Users/ava/Downloads/iopddl
# Build directory: /Users/ava/Downloads/iopddl/build_2
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[iopddl_test]=] "/Users/ava/Downloads/iopddl/build_2/iopddl_test")
set_tests_properties([=[iopddl_test]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/ava/Downloads/iopddl/CMakeLists.txt;53;add_test;/Users/ava/Downloads/iopddl/CMakeLists.txt;0;")
subdirs("external/abseil-cpp")
subdirs("external/googletest")
subdirs("external/nlohmann_json")
