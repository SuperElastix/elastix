/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#include "itkOpenCLStringUtils.h"

int
main( int argc, char * argv[] )
{
  // Test itk::opencl_simplified()
  const std::string str_simplified_input1           = "   ITK\t OpenCL\nRock \tand \r Roll!\r\n ";
  const std::string str_simplified_expected_output1 = "ITK OpenCL Rock and Roll!";

  if( str_simplified_expected_output1 != itk::opencl_simplified( str_simplified_input1 ) )
  {
    return EXIT_FAILURE;
  }

  const std::string str_simplified_input2
    =
    "cl_khr_byte_addressable_store cl_khr_icd cl_khr_gl_sharing cl_nv_d3d9_sharing cl_nv_d3d10_sharing cl_khr_d3d10_sharing cl_nv_d3d11_sharing cl_nv_compiler_options cl_nv_device_attribute_query cl_nv_pragma_unroll  cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_khr_fp64 ";
  const std::string str_simplified_expected_output2
    =
    "cl_khr_byte_addressable_store cl_khr_icd cl_khr_gl_sharing cl_nv_d3d9_sharing cl_nv_d3d10_sharing cl_khr_d3d10_sharing cl_nv_d3d11_sharing cl_nv_compiler_options cl_nv_device_attribute_query cl_nv_pragma_unroll cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics cl_khr_local_int32_base_atomics cl_khr_local_int32_extended_atomics cl_khr_fp64";

  const std::string str_simplified_output2 = itk::opencl_simplified( str_simplified_input2 );

  if( str_simplified_expected_output2 != str_simplified_output2 )
  {
    return EXIT_FAILURE;
  }

  const std::list< std::string > res_split1 = itk::opencl_split_string( str_simplified_output2, ' ' );

  if( res_split1.size() != 15 )
  {
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
