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
#ifndef itkOpenCLContextScopeGuard_h
#define itkOpenCLContextScopeGuard_h

#include "itkMacro.h" // For ITK_DISALLOW_COPY_AND_MOVE
#include "itkTestHelper.h"

#include <exception>
#include <iostream> // For cerr.

namespace itk
{

/** Releases the global OpenCL context when the guard gets out of scope. */
class OpenCLContextScopeGuard
{
public:
  OpenCLContextScopeGuard() = default;
  ITK_DISALLOW_COPY_AND_MOVE(OpenCLContextScopeGuard);

  ~OpenCLContextScopeGuard()
  {
    try
    {
      ReleaseContext();
    }
    catch (const std::exception & stdException)
    {
      std::cerr << "ReleaseContext() failed.\nException: " << stdException.what() << '\n';
    }
  }
};

} // namespace itk

#endif
