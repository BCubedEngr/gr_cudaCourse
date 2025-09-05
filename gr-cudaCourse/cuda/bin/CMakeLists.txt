SET(bin_files check_gpu.cu)
SET(install_files "")

foreach(bin_file IN ITEMS ${bin_files})
    get_filename_component(name ${bin_file} NAME_WE)
    add_executable(${name} ${bin_file})
    # We don't want to use the CUDA compiler here, because we need to be able to
    # link with just gcc
    set_target_properties(${name} PROPERTIES LINKER_LANGUAGE CXX)
    target_link_libraries(${name} gr_cuda_course)
    LIST(APPEND install_files ${PROJECT_BINARY_DIR}/bin/${name})
endforeach()

INSTALL(FILES ${install_files} DESTINATION bin
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

