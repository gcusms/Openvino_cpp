# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

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
CMAKE_COMMAND = /usr/local/lib/python3.8/dist-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /usr/local/lib/python3.8/dist-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/wolf/Desktop/temp2/demo

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/wolf/Desktop/temp2/demo/build

# Include any dependencies generated for this target.
include CMakeFiles/Camera.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/Camera.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/Camera.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Camera.dir/flags.make

CMakeFiles/Camera.dir/devices/camera/mv_video_capture.cpp.o: CMakeFiles/Camera.dir/flags.make
CMakeFiles/Camera.dir/devices/camera/mv_video_capture.cpp.o: ../devices/camera/mv_video_capture.cpp
CMakeFiles/Camera.dir/devices/camera/mv_video_capture.cpp.o: CMakeFiles/Camera.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wolf/Desktop/temp2/demo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Camera.dir/devices/camera/mv_video_capture.cpp.o"
	/bin/x86_64-linux-gnu-g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/Camera.dir/devices/camera/mv_video_capture.cpp.o -MF CMakeFiles/Camera.dir/devices/camera/mv_video_capture.cpp.o.d -o CMakeFiles/Camera.dir/devices/camera/mv_video_capture.cpp.o -c /home/wolf/Desktop/temp2/demo/devices/camera/mv_video_capture.cpp

CMakeFiles/Camera.dir/devices/camera/mv_video_capture.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Camera.dir/devices/camera/mv_video_capture.cpp.i"
	/bin/x86_64-linux-gnu-g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wolf/Desktop/temp2/demo/devices/camera/mv_video_capture.cpp > CMakeFiles/Camera.dir/devices/camera/mv_video_capture.cpp.i

CMakeFiles/Camera.dir/devices/camera/mv_video_capture.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Camera.dir/devices/camera/mv_video_capture.cpp.s"
	/bin/x86_64-linux-gnu-g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wolf/Desktop/temp2/demo/devices/camera/mv_video_capture.cpp -o CMakeFiles/Camera.dir/devices/camera/mv_video_capture.cpp.s

# Object files for target Camera
Camera_OBJECTS = \
"CMakeFiles/Camera.dir/devices/camera/mv_video_capture.cpp.o"

# External object files for target Camera
Camera_EXTERNAL_OBJECTS =

libCamera.a: CMakeFiles/Camera.dir/devices/camera/mv_video_capture.cpp.o
libCamera.a: CMakeFiles/Camera.dir/build.make
libCamera.a: CMakeFiles/Camera.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/wolf/Desktop/temp2/demo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libCamera.a"
	$(CMAKE_COMMAND) -P CMakeFiles/Camera.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Camera.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Camera.dir/build: libCamera.a
.PHONY : CMakeFiles/Camera.dir/build

CMakeFiles/Camera.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Camera.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Camera.dir/clean

CMakeFiles/Camera.dir/depend:
	cd /home/wolf/Desktop/temp2/demo/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/wolf/Desktop/temp2/demo /home/wolf/Desktop/temp2/demo /home/wolf/Desktop/temp2/demo/build /home/wolf/Desktop/temp2/demo/build /home/wolf/Desktop/temp2/demo/build/CMakeFiles/Camera.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Camera.dir/depend

