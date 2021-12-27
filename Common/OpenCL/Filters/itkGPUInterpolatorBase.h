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
#ifndef itkGPUInterpolatorBase_h
#define itkGPUInterpolatorBase_h

#include "itkGPUDataManager.h"

namespace itk
{
/** \class GPUInterpolatorBase
 * \brief Base class fro all GPU interpolators.
 *
 * \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands
 *
 * \note This work was funded by the Netherlands Organisation for
 * Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
 *
 * \ingroup GPUCommon
 */
class ITK_EXPORT GPUInterpolatorBase
{
public:
  /** Run-time type information (and related methods). */
  virtual const char *
  GetNameOfClass() const
  {
    return "GPUInterpolatorBase";
  }

  /** Returns OpenCL \a source code for the transform.
   * Returns true if source code was combined, false otherwise. */
  virtual bool
  GetSourceCode(std::string & source) const;

  /** Returns data manager that stores all settings for the transform. */
  virtual GPUDataManager::Pointer
  GetParametersDataManager() const;

protected:
  GPUInterpolatorBase();
  virtual ~GPUInterpolatorBase() = default;

  GPUDataManager::Pointer m_ParametersDataManager;
};

} // end namespace itk

#endif /* itkGPUInterpolatorBase_h */
