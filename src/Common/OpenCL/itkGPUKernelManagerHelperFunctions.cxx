/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#include "itkGPUKernelManagerHelperFunctions.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

namespace itk
{
//----------------------------------------------------------------------------
bool LoadProgramFromFile( const std::string & _filename, std::string & _source,
                          const bool skipHeader )
{
  const std::size_t headerSize = 572;
  std::ifstream     fileStream( _filename.c_str() );

  if ( fileStream.fail() )
  {
    itkGenericExceptionMacro( << "Unable to open file: " << _filename );
    fileStream.close();
    return false;
  }

  std::stringstream oss;
  if ( skipHeader )
  {
    fileStream.seekg( headerSize, std::ios::beg );
  }

  oss << fileStream.rdbuf();

  if ( !fileStream && !fileStream.eof() )
  {
    itkGenericExceptionMacro( << "Error reading file: " << _filename );
    fileStream.close();
    return false;
  }

  _source = oss.str();

  return true;
}

//----------------------------------------------------------------------------
bool LoadProgramFromFile( const std::string & _filename,
                          std::vector< std::string > & _sources,
                          const std::string & _name,
                          const bool skipHeader )
{
  bool        sourceLoaded;
  std::string source;

  if ( LoadProgramFromFile( _filename, source, skipHeader ) )
  {
    sourceLoaded = true;
    _sources.push_back( source );
  }
  else
  {
    itkGenericExceptionMacro( << _name << " has not been loaded from: " << _filename );
    sourceLoaded = false;
  }

  return sourceLoaded;
}
} // end namespace itk
