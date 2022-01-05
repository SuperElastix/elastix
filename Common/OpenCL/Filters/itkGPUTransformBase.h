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
#ifndef itkGPUTransformBase_h
#define itkGPUTransformBase_h

#include "itkGPUDataManager.h"

namespace itk
{
/** \class GPUTransformBase
 * \brief Base class for all GPU transforms.
 *
 * \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands
 *
 * \note This work was funded by the Netherlands Organisation for
 * Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
 *
 * \ingroup GPUCommon
 */
class ITK_EXPORT GPUTransformBase
{
public:
  /** Standard class typedefs. */
  using Self = GPUTransformBase;

  /** Run-time type information (and related methods). */
  virtual const char *
  GetNameOfClass() const
  {
    return "GPUTransformBase";
  }

  /** Returns OpenCL \a source code for the transform.
   * Returns true if source code was combined, false otherwise. */
  virtual bool
  GetSourceCode(std::string & source) const;

  /** Returns true if the derived transform is identity transform,
   * false otherwise. */
  virtual bool
  IsIdentityTransform() const
  {
    return false;
  }

  /** Returns true if the derived transform is matrix offset transform,
   * false otherwise. */
  virtual bool
  IsMatrixOffsetTransform() const
  {
    return false;
  }

  /** Returns true if the derived transform is translation transform,
   * false otherwise. */
  virtual bool
  IsTranslationTransform() const
  {
    return false;
  }

  /** Returns true if the derived transform is BSpline transform,
   * false otherwise. */
  virtual bool
  IsBSplineTransform() const
  {
    return false;
  }

  /** Returns data manager that stores all settings for the transform. */
  virtual GPUDataManager::Pointer
  GetParametersDataManager() const;

  /** Returns data manager that stores all settings for the transform \a index.
   * Used by combination transforms. */
  virtual GPUDataManager::Pointer
  GetParametersDataManager(const std::size_t index) const;

protected:
  GPUTransformBase();
  virtual ~GPUTransformBase() = default;

  GPUDataManager::Pointer m_ParametersDataManager;

private:
  GPUTransformBase(const Self & other) = delete;
  const Self &
  operator=(const Self &) = delete;
};

} // end namespace itk

#endif /* itkGPUTransformBase_h */
