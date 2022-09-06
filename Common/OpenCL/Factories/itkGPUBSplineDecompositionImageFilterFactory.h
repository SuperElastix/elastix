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
#ifndef itkGPUBSplineDecompositionImageFilterFactory_h
#define itkGPUBSplineDecompositionImageFilterFactory_h

#include "itkGPUObjectFactoryBase.h"
#include "itkGPUBSplineDecompositionImageFilter.h"

namespace itk
{
/** \class GPUBSplineDecompositionImageFilterFactory2
 * \brief Object Factory implementation for GPUBSplineDecompositionImageFilter
 *
 * \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands
 *
 * \note This work was funded by the Netherlands Organisation for
 * Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
 *
 */
template <typename TTypeListIn, typename TTypeListOut, typename NDimensions>
class ITK_TEMPLATE_EXPORT GPUBSplineDecompositionImageFilterFactory2 : public GPUObjectFactoryBase<NDimensions>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(GPUBSplineDecompositionImageFilterFactory2);

  using Self = GPUBSplineDecompositionImageFilterFactory2;
  using Superclass = GPUObjectFactoryBase<NDimensions>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Return a descriptive string describing the factory. */
  const char *
  GetDescription() const
  {
    return "A Factory for GPUBSplineDecompositionImageFilter";
  }

  /** Method for class instantiation. */
  itkFactorylessNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(GPUBSplineDecompositionImageFilterFactory2, GPUObjectFactoryBase);

  /** Register one factory of this type. */
  static void
  RegisterOneFactory();

  /** Operator() to register override. */
  template <typename TTypeIn, typename TTypeOut, unsigned int VImageDimension>
  void
  operator()()
  {
    // Image typedefs
    using InputImageType = Image<TTypeIn, VImageDimension>;
    using OutputImageType = Image<TTypeOut, VImageDimension>;
    using GPUInputImageType = GPUImage<TTypeIn, VImageDimension>;
    using GPUOutputImageType = GPUImage<TTypeOut, VImageDimension>;

    // Override default
    this->RegisterOverride(
      typeid(BSplineDecompositionImageFilter<InputImageType, OutputImageType>).name(),
      typeid(GPUBSplineDecompositionImageFilter<InputImageType, OutputImageType>).name(),
      "GPU BSplineDecompositionImageFilter override default",
      true,
      CreateObjectFunction<GPUBSplineDecompositionImageFilter<InputImageType, OutputImageType>>::New());

    // Override when itkGPUImage is first template argument
    this->RegisterOverride(
      typeid(BSplineDecompositionImageFilter<GPUInputImageType, OutputImageType>).name(),
      typeid(GPUBSplineDecompositionImageFilter<GPUInputImageType, OutputImageType>).name(),
      "GPU BSplineDecompositionImageFilter override GPUImage first",
      true,
      CreateObjectFunction<GPUBSplineDecompositionImageFilter<GPUInputImageType, OutputImageType>>::New());

    // Override when itkGPUImage is second template argument
    this->RegisterOverride(
      typeid(BSplineDecompositionImageFilter<InputImageType, GPUOutputImageType>).name(),
      typeid(GPUBSplineDecompositionImageFilter<InputImageType, GPUOutputImageType>).name(),
      "GPU BSplineDecompositionImageFilter override GPUImage second",
      true,
      CreateObjectFunction<GPUBSplineDecompositionImageFilter<InputImageType, GPUOutputImageType>>::New());

    // Override when itkGPUImage is first and second template arguments
    this->RegisterOverride(
      typeid(BSplineDecompositionImageFilter<GPUInputImageType, GPUOutputImageType>).name(),
      typeid(GPUBSplineDecompositionImageFilter<GPUInputImageType, GPUOutputImageType>).name(),
      "GPU BSplineDecompositionImageFilter override GPUImage first and second",
      true,
      CreateObjectFunction<GPUBSplineDecompositionImageFilter<GPUInputImageType, GPUOutputImageType>>::New());
  }


protected:
  GPUBSplineDecompositionImageFilterFactory2();
  virtual ~GPUBSplineDecompositionImageFilterFactory2() {}

  /** Register methods for 1D. */
  virtual void
  Register1D();

  /** Register methods for 2D. */
  virtual void
  Register2D();

  /** Register methods for 3D. */
  virtual void
  Register3D();
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkGPUBSplineDecompositionImageFilterFactory.hxx"
#endif

#endif // end #ifndef itkGPUBSplineDecompositionImageFilterFactory_h
