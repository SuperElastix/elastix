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
#ifndef itkGPUBSplineInterpolateImageFunctionFactory_h
#define itkGPUBSplineInterpolateImageFunctionFactory_h

#include "itkGPUObjectFactoryBase.h"
#include "itkGPUBSplineInterpolateImageFunction.h"

namespace itk
{
/** \class GPUBSplineInterpolateImageFunctionFactory2
 * \brief Object Factory implementation for GPUBSplineInterpolateImageFunction
 *
 * \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands
 *
 * \note This work was funded by the Netherlands Organisation for
 * Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
 *
 */
template <typename TTypeList, typename NDimensions>
class ITK_TEMPLATE_EXPORT GPUBSplineInterpolateImageFunctionFactory2 : public GPUObjectFactoryBase<NDimensions>
{
public:
  typedef GPUBSplineInterpolateImageFunctionFactory2 Self;
  typedef GPUObjectFactoryBase<NDimensions>          Superclass;
  typedef SmartPointer<Self>                         Pointer;
  typedef SmartPointer<const Self>                   ConstPointer;

  /** Return a descriptive string describing the factory. */
  const char *
  GetDescription() const
  {
    return "A Factory for GPUBSplineInterpolateImageFunction";
  }

  /** Method for class instantiation. */
  itkFactorylessNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(GPUBSplineInterpolateImageFunctionFactory2, GPUObjectFactoryBase);

  /** Register one factory of this type. */
  static void
  RegisterOneFactory();

  /** Operator() to register override. */
  template <typename TType, unsigned int VImageDimension>
  void
  operator()(void)
  {
    // Image typedefs
    typedef Image<TType, VImageDimension>    InputImageType;
    typedef GPUImage<TType, VImageDimension> GPUInputImageType;

    // Override default with the coordinate representation type as float
    // and coefficient type as float
    this->RegisterOverride(
      typeid(BSplineInterpolateImageFunction<InputImageType, float, float>).name(),
      typeid(GPUBSplineInterpolateImageFunction<InputImageType, float, float>).name(),
      "GPU BSplineInterpolateImageFunction override with coord rep and coefficient as float",
      true,
      CreateObjectFunction<GPUBSplineInterpolateImageFunction<InputImageType, float, float>>::New());

    // Override when itkGPUImage is first template argument,
    // the coordinate representation type as float and coefficient type as float
    this->RegisterOverride(
      typeid(BSplineInterpolateImageFunction<GPUInputImageType, float, float>).name(),
      typeid(GPUBSplineInterpolateImageFunction<GPUInputImageType, float, float>).name(),
      "GPU BSplineInterpolateImageFunction override for GPUImage with coord rep and coefficient as float",
      true,
      CreateObjectFunction<GPUBSplineInterpolateImageFunction<GPUInputImageType, float, float>>::New());

    // Override default with and the coordinate representation type as double
    // and coefficient type as double
    this->RegisterOverride(
      typeid(BSplineInterpolateImageFunction<InputImageType, double, double>).name(),
      typeid(GPUBSplineInterpolateImageFunction<InputImageType, double, double>).name(),
      "GPU BSplineInterpolateImageFunction override with coord rep and coefficient as double",
      true,
      CreateObjectFunction<GPUBSplineInterpolateImageFunction<InputImageType, double, double>>::New());

    // Override when itkGPUImage is first template argument,
    // the coordinate representation type as double and coefficient type as double
    this->RegisterOverride(
      typeid(BSplineInterpolateImageFunction<GPUInputImageType, double, double>).name(),
      typeid(GPUBSplineInterpolateImageFunction<GPUInputImageType, double, double>).name(),
      "GPU BSplineInterpolateImageFunction override for GPUImage with coord rep and coefficient as double",
      true,
      CreateObjectFunction<GPUBSplineInterpolateImageFunction<GPUInputImageType, double, double>>::New());
  }


protected:
  GPUBSplineInterpolateImageFunctionFactory2();
  virtual ~GPUBSplineInterpolateImageFunctionFactory2() {}

  /** Register methods for 1D. */
  virtual void
  Register1D();

  /** Register methods for 2D. */
  virtual void
  Register2D();

  /** Register methods for 3D. */
  virtual void
  Register3D();

private:
  GPUBSplineInterpolateImageFunctionFactory2(const Self &) = delete;
  void
  operator=(const Self &) = delete;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkGPUBSplineInterpolateImageFunctionFactory.hxx"
#endif

#endif // end #ifndef itkGPUBSplineInterpolateImageFunctionFactory_h
