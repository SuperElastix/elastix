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
#ifndef itkGPUAdvancedEuler2DTransformFactory_h
#define itkGPUAdvancedEuler2DTransformFactory_h

#include "itkGPUObjectFactoryBase.h"
#include "itkGPUAdvancedEuler2DTransform.h"

namespace itk
{
/** \class GPUAdvancedEuler2DTransformFactory
 * \brief Object Factory implementation for GPUAdvancedEuler2DTransform
 *
 * \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands
 *
 * \note This work was funded by the Netherlands Organisation for
 * Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
 *
 */
template <typename NDimensions>
class ITK_TEMPLATE_EXPORT GPUAdvancedEuler2DTransformFactory2 : public GPUObjectFactoryBase<NDimensions>
{
public:
  typedef GPUAdvancedEuler2DTransformFactory2 Self;
  typedef GPUObjectFactoryBase<NDimensions>   Superclass;
  typedef SmartPointer<Self>                  Pointer;
  typedef SmartPointer<const Self>            ConstPointer;

  /** Return a descriptive string describing the factory. */
  const char *
  GetDescription() const
  {
    return "A Factory for GPUAdvancedEuler2DTransform";
  }

  /** Method for class instantiation. */
  itkFactorylessNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(GPUAdvancedEuler2DTransformFactory2, GPUObjectFactoryBase);

  /** Register one factory of this type. */
  static void
  RegisterOneFactory();

  /** Operator() to register override. */
  template <typename TType>
  void
  operator()(void)
  {
    this->RegisterOverride(typeid(AdvancedEuler2DTransform<TType>).name(),
                           typeid(GPUAdvancedEuler2DTransform<TType>).name(),
                           "GPU AdvancedEuler2DTransform override",
                           true,
                           CreateObjectFunction<GPUAdvancedEuler2DTransform<TType>>::New());
  }


protected:
  GPUAdvancedEuler2DTransformFactory2();
  virtual ~GPUAdvancedEuler2DTransformFactory2() {}

  /** Typedef for real type list. */
  typedef typelist::MakeTypeList<float, double>::Type RealTypeList;

  /** Register methods for 2D. */
  virtual void
  Register2D();

private:
  GPUAdvancedEuler2DTransformFactory2(const Self &) = delete;
  void
  operator=(const Self &) = delete;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkGPUAdvancedEuler2DTransformFactory.hxx"
#endif

#endif /* itkGPUAdvancedEuler2DTransformFactory_h */
