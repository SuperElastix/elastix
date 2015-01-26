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
/*=========================================================================
*
*  Copyright Insight Software Consortium
*
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*         http://www.apache.org/licenses/LICENSE-2.0.txt
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*
*=========================================================================*/
#include "itkOpenCLUtil.h"
#include <itkVector.h>

namespace itk
{
//------------------------------------------------------------------------------
//
// Get the block size based on the desired image dimension
//
int
OpenCLGetLocalBlockSize( unsigned int ImageDim )
{
  /**
  * OpenCL workgroup (block) size for 1/2/3D - needs to be tuned based on the GPU architecture
  * 1D : 256
  * 2D : 16x16 = 256
  * 3D : 4x4x4 = 64
  */

  if( ImageDim > 3 || ImageDim < 1 )
  {
    itkGenericExceptionMacro( "Only ImageDimensions up to 3 are supported" );
  }

#ifdef OPENCL_USE_INTEL
  // let's just return 1 for Intel, that is safe. will fix it later.
  return 1;
#endif

  int OPENCL_BLOCK_SIZE[ 3 ] = { 256, 16, 4 /*8*/ };
  return OPENCL_BLOCK_SIZE[ ImageDim - 1 ];
}


//------------------------------------------------------------------------------
std::string
GetTypename( const std::type_info & intype )
{
  std::string typestr;

  if( intype == typeid( unsigned char )
    || intype == typeid( itk::Vector< unsigned char, 2 > )
    || intype == typeid( itk::Vector< unsigned char, 3 > ) )
  {
    typestr = "unsigned char";
  }
  else if( intype == typeid( char )
    || intype == typeid( itk::Vector< char, 2 > )
    || intype == typeid( itk::Vector< char, 3 > ) )
  {
    typestr = "char";
  }
  else if( intype == typeid( short )
    || intype == typeid( itk::Vector< short, 2 > )
    || intype == typeid( itk::Vector< short, 3 > ) )
  {
    typestr = "short";
  }
  else if( intype == typeid( unsigned short )
    || intype == typeid( itk::Vector< unsigned short, 2 > )
    || intype == typeid( itk::Vector< unsigned short, 3 > ) )
  {
    typestr = "unsigned short";
  }
  else if( intype == typeid( int )
    || intype == typeid( itk::Vector< int, 2 > )
    || intype == typeid( itk::Vector< int, 3 > ) )
  {
    typestr = "int";
  }
  else if( intype == typeid( unsigned int )
    || intype == typeid( itk::Vector< unsigned int, 2 > )
    || intype == typeid( itk::Vector< unsigned int, 3 > ) )
  {
    typestr = "unsigned int";
  }
  else if( intype == typeid( long )
    || intype == typeid( itk::Vector< long, 2 > )
    || intype == typeid( itk::Vector< long, 3 > ) )
  {
    typestr = "long";
  }
  else if( intype == typeid( unsigned long )
    || intype == typeid( itk::Vector< unsigned long, 2 > )
    || intype == typeid( itk::Vector< unsigned long, 3 > ) )
  {
    typestr = "unsigned long";
  }
  else if( intype == typeid( long long )
    || intype == typeid( itk::Vector< long long, 2 > )
    || intype == typeid( itk::Vector< long long, 3 > ) )
  {
    typestr = "long long";
  }
  else if( intype == typeid( float )
    || intype == typeid( itk::Vector< float, 2 > )
    || intype == typeid( itk::Vector< float, 3 > ) )
  {
    typestr = "float";
  }
  else if( intype == typeid( double )
    || intype == typeid( itk::Vector< double, 2 > )
    || intype == typeid( itk::Vector< double, 3 > ) )
  {
    typestr = "double";
  }
  else
  {
    itkGenericExceptionMacro( "Unknown type: " << intype.name() );
  }
  return typestr;
}


//------------------------------------------------------------------------------
/** Get 64-bit pragma */
std::string
Get64BitPragma()
{
  std::ostringstream msg;

  msg << "#ifdef cl_khr_fp64\n";
  msg << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
  msg << "#endif\n";
  msg << "#ifdef cl_amd_fp64\n";
  msg << "#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n";
  msg << "#endif\n";
  return msg.str();
}


//------------------------------------------------------------------------------
void
GetTypenameInString( const std::type_info & intype, std::ostringstream & ret )
{
  const std::string typestr = GetTypename( intype );
  ret << typestr << "\n";
  if( typestr == "double" )
  {
    std::string pragmastr = Get64BitPragma();
    ret << pragmastr;
  }
}


} // end of namespace itk
