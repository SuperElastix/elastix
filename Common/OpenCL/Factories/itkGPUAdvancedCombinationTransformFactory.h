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
#ifndef itkGPUAdvancedCombinationTransformFactory_h
#define itkGPUAdvancedCombinationTransformFactory_h

#include "itkGPUObjectFactoryBase.h"
#include "itkGPUAdvancedCombinationTransform.h"

namespace itk
{
/** \class GPUAdvancedCombinationTransformFactory
 * \brief Object Factory implementation for GPUAdvancedCombinationTransform
 *
 * \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands
 *
 * \note This work was funded by the Netherlands Organisation for
 * Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
 *
 */
template <typename NDimensions>
class ITK_TEMPLATE_EXPORT GPUAdvancedCombinationTransformFactory2 : public GPUObjectFactoryBase<NDimensions>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(GPUAdvancedCombinationTransformFactory2);

  using Self = GPUAdvancedCombinationTransformFactory2;
  using Superclass = GPUObjectFactoryBase<NDimensions>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Return a descriptive string describing the factory. */
  const char *
  GetDescription() const
  {
    return "A Factory for GPUAdvancedCombinationTransform";
  }

  /** Method for class instantiation. */
  itkFactorylessNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(GPUAdvancedCombinationTransformFactory2);

  /** Register one factory of this type. */
  static void
  RegisterOneFactory();

  /** Operator() to register override. */
  template <typename TType, unsigned int VImageDimension>
  void
  operator()()
  {
    this->RegisterOverride(typeid(AdvancedCombinationTransform<TType, VImageDimension>).name(),
                           typeid(GPUAdvancedCombinationTransform<TType, VImageDimension>).name(),
                           "GPU AdvancedCombinationTransformFactory override",
                           true,
                           CreateObjectFunction<GPUAdvancedCombinationTransform<TType, VImageDimension>>::New());
  }


protected:
  GPUAdvancedCombinationTransformFactory2();
  virtual ~GPUAdvancedCombinationTransformFactory2() {}

  /** Typedef for real type list. */
  using RealTypeList = typelist::MakeTypeList<float, double>::Type;

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
#  include "itkGPUAdvancedCombinationTransformFactory.hxx"
#endif

#endif /* itkGPUAdvancedCombinationTransformFactory_h */
