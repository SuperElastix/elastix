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
#ifndef itkGPUEuler2DTransformFactory_h
#define itkGPUEuler2DTransformFactory_h

#include "itkGPUObjectFactoryBase.h"
#include "itkGPUEuler2DTransform.h"

namespace itk
{
/** \class GPUEuler2DTransformFactory
 * \brief Object Factory implementation for GPUEuler2DTransform
 *
 * \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands
 *
 * \note This work was funded by the Netherlands Organisation for
 * Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
 *
 */
template <typename NDimensions>
class ITK_TEMPLATE_EXPORT GPUEuler2DTransformFactory2 : public GPUObjectFactoryBase<NDimensions>
{
public:
  typedef GPUEuler2DTransformFactory2       Self;
  typedef GPUObjectFactoryBase<NDimensions> Superclass;
  typedef SmartPointer<Self>                Pointer;
  typedef SmartPointer<const Self>          ConstPointer;

  /** Return a descriptive string describing the factory. */
  const char *
  GetDescription() const
  {
    return "A Factory for GPUEuler2DTransform";
  }

  /** Method for class instantiation. */
  itkFactorylessNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(GPUEuler2DTransformFactory2, GPUObjectFactoryBase);

  /** Register one factory of this type. */
  static void
  RegisterOneFactory();

  /** Operator() to register override. */
  template <typename TType>
  void
  operator()(void)
  {
    this->RegisterOverride(typeid(Euler2DTransform<TType>).name(),
                           typeid(GPUEuler2DTransform<TType>).name(),
                           "GPU Euler2DTransform override",
                           true,
                           CreateObjectFunction<GPUEuler2DTransform<TType>>::New());
  }


protected:
  GPUEuler2DTransformFactory2();
  virtual ~GPUEuler2DTransformFactory2() {}

  /** Typedef for real type list. */
  typedef typelist::MakeTypeList<float, double>::Type RealTypeList;

  /** Register methods for 2D. */
  virtual void
  Register2D();

private:
  GPUEuler2DTransformFactory2(const Self &) = delete;
  void
  operator=(const Self &) = delete;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkGPUEuler2DTransformFactory.hxx"
#endif

#endif /* itkGPUEuler2DTransformFactory_h */
