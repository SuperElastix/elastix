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
#ifndef itkGPUInterpolateImageFunction_h
#define itkGPUInterpolateImageFunction_h

#include "itkInterpolateImageFunction.h"
#include "itkGPUInterpolatorBase.h"

namespace itk
{
/** \class GPUInterpolateImageFunction
 * \brief GPU version of InterpolateImageFunction.
 *
 * \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands
 *
 * \note This work was funded by the Netherlands Organisation for
 * Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
 *
 * \ingroup GPUCommon
 */
template <typename TInputImage,
          typename TCoordRep = float,
          typename TParentInterpolateImageFunction = InterpolateImageFunction<TInputImage, TCoordRep>>
class ITK_TEMPLATE_EXPORT GPUInterpolateImageFunction
  : public TParentInterpolateImageFunction
  , public GPUInterpolatorBase
{
public:
  /** Standard class typedefs. */
  typedef GPUInterpolateImageFunction     Self;
  typedef TParentInterpolateImageFunction CPUSuperclass;
  typedef GPUInterpolatorBase             GPUSuperclass;
  typedef SmartPointer<Self>              Pointer;
  typedef SmartPointer<const Self>        ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro(GPUInterpolateImageFunction, TParentInterpolateImageFunction);

  /** ImageDimension constants */
  itkStaticConstMacro(InputImageDimension, unsigned int, TInputImage::ImageDimension);

  /** Superclass typedef support. */
  typedef typename CPUSuperclass::InputImageType      InputImageType;
  typedef typename CPUSuperclass::ContinuousIndexType ContinuousIndexType;
  typedef typename CPUSuperclass::CoordRepType        CoordRepType;

protected:
  GPUInterpolateImageFunction();
  ~GPUInterpolateImageFunction() override = default;
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  /** Returns data manager that stores all settings for the transform. */
  GPUDataManager::Pointer
  GetParametersDataManager(void) const override;

private:
  GPUInterpolateImageFunction(const Self &) = delete;
  void
  operator=(const Self &) = delete;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkGPUInterpolateImageFunction.hxx"
#endif

#endif /* itkGPUInterpolateImageFunction_h */
