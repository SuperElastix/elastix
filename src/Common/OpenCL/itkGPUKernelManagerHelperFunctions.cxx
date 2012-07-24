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

#include "itkGPUKernelManagerHelperFunctions.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

namespace itk
{
  //----------------------------------------------------------------------------
  bool LoadProgramFromFile(const std::string &_filename, std::string &_source,
    const bool skipHeader)
  {
    const std::size_t headerSize = 760;
    std::ifstream fileStream(_filename.c_str());
    if (fileStream.fail())
    {
      itkGenericExceptionMacro(<< "Unable to open file: "<< _filename);
      fileStream.close();
      return false;
    }

    std::stringstream oss;
    if(skipHeader)
      fileStream.seekg(headerSize, std::ios::beg);

    oss << fileStream.rdbuf();

    if(!fileStream && !fileStream.eof())
    {
      itkGenericExceptionMacro(<< "Error reading file: "<< _filename);
      fileStream.close();
      return false;
    }

    _source = oss.str();

    return true;
  }

  //----------------------------------------------------------------------------
  bool LoadProgramFromFile(const std::string &_filename,
    std::vector<std::string> &_sources, const std::string &_name,
    const bool skipHeader)
  {
    bool sourceLoaded;
    std::string source;
    if(LoadProgramFromFile(_filename, source, skipHeader))
    {
      sourceLoaded = true;
      _sources.push_back(source);
    }
    else
    {
      itkGenericExceptionMacro( << _name << " has not been loaded from: " << _filename );
      sourceLoaded = false;
    }

    return sourceLoaded;
  }
}
