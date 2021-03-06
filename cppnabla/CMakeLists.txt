set( LIBRARY_NAME "cppnabla" )

set( KOKKOS_INSTALL_PATH "Undef" CACHE STRING "Path to the Kokkos installation directory" )

message( STATUS "KOKKOS_INSTALL_PATH is ${KOKKOS_INSTALL_PATH}" )
if( ${KOKKOS_INSTALL_PATH} STREQUAL "Undef" )
    message( FATAL_ERROR "Please specify the KOKKOS installation directory (-DKOKKOS_INSTALL_PATH)" )
endif()

list( APPEND CMAKE_PREFIX_PATH "${KOKKOS_INSTALL_PATH}" )
find_package( Kokkos REQUIRED )

# pour lever (?) l'erreur
# non-type template argument evaluates to 2, which cannot be narrowed to type 'bool'
# dans Multiarray.h
if(CMAKE_CXX_COMPILER_ID STREQUAL Clang OR
      CMAKE_CXX_COMPILER_ID STREQUAL AppleClang)
    string(APPEND CMAKE_CXX_FLAGS " -Wno-c++11-narrowing")
endif()
# mais donne une autre erreur
# use of overloaded operator '<<' is ambiguous (with operand types 'basic_ostream
# dans Multiarray.c

if( ${Kokkos_FOUND} )
    message( STATUS "Kokkos found" )
endif()

add_library( ${LIBRARY_NAME} )

target_sources( ${LIBRARY_NAME} PRIVATE
    "types/MultiArray.cc"
    "mesh/CartesianMesh2D.cc"
    "mesh/CartesianMesh2DGenerator.cc"
    "mesh/FileWriter.cc"
    "mesh/VtkFileWriter2D.cc"
    "mesh/PvdFileWriter2D.cc"
    "utils/Utils.cc"
    "utils/Timer.cc"
)

target_include_directories( ${LIBRARY_NAME} PUBLIC ${CMAKE_CURRENT_LIST_DIR} )
target_link_libraries( ${LIBRARY_NAME}
PUBLIC
    Kokkos::kokkos
)
