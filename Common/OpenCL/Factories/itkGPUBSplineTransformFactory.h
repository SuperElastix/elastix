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
#ifndef itkGPUBSplineTransformFactory_h
#define itkGPUBSplineTransformFactory_h

#include "itkGPUObjectFactoryBase.h"
#include "itkGPUBSplineTransform.h"

namespace itk
{
/** \class GPUBSplineTransformFactory
 * \brief Object Factory implementation for GPUBSplineTransform
 *
 * \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands
 *
 * \note This work was funded by the Netherlands Organisation for
 * Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
 *
 */
template <typename NDimensions>
class ITK_TEMPLATE_EXPORT GPUBSplineTransformFactory2 : public GPUObjectFactoryBase<NDimensions>
{
public:
  typedef GPUBSplineTransformFactory2       Self;
  typedef GPUObjectFactoryBase<NDimensions> Superclass;
  typedef SmartPointer<Self>                Pointer;
  typedef SmartPointer<const Self>          ConstPointer;

  /** Return a descriptive string describing the factory. */
  const char *
  GetDescription() const
  {
    return "A Factory for GPUBSplineTransform";
  }

  /** Method for class instantiation. */
  itkFactorylessNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(GPUBSplineTransformFactory2, GPUObjectFactoryBase);

  /** Register one factory of this type. */
  static void
  RegisterOneFactory();

  /** Operator() to register override. */
  template <typename TType, unsigned int VImageDimension>
  void
  operator()(void)
  {
    // Override for spline order equal 1
    this->RegisterOverride(typeid(BSplineTransform<TType, VImageDimension, 1>).name(),
                           typeid(GPUBSplineTransform<TType, VImageDimension, 1>).name(),
                           "GPU BSplineTransform override",
                           true,
                           CreateObjectFunction<GPUBSplineTransform<TType, VImageDimension, 1>>::New());

    // Override for spline order equal 2
    this->RegisterOverride(typeid(BSplineTransform<TType, VImageDimension, 2>).name(),
                           typeid(GPUBSplineTransform<TType, VImageDimension, 2>).name(),
                           "GPU BSplineTransform override",
                           true,
                           CreateObjectFunction<GPUBSplineTransform<TType, VImageDimension, 2>>::New());

    // Override for spline order equal 3
    this->RegisterOverride(typeid(BSplineTransform<TType, VImageDimension, 3>).name(),
                           typeid(GPUBSplineTransform<TType, VImageDimension, 3>).name(),
                           "GPU BSplineTransform override",
                           true,
                           CreateObjectFunction<GPUBSplineTransform<TType, VImageDimension, 3>>::New());
  }


protected:
  GPUBSplineTransformFactory2();
  virtual ~GPUBSplineTransformFactory2() {}

  /** Typedef for real type list. */
  typedef typelist::MakeTypeList<float, double>::Type RealTypeList;

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
  GPUBSplineTransformFactory2(const Self &) = delete;
  void
  operator=(const Self &) = delete;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkGPUBSplineTransformFactory.hxx"
#endif

#endif /* itkGPUBSplineTransformFactory_h */
