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
#ifndef itkGPUSimilarity2DTransformFactory_h
#define itkGPUSimilarity2DTransformFactory_h

#include "itkGPUObjectFactoryBase.h"
#include "itkGPUSimilarity2DTransform.h"

namespace itk
{
/** \class GPUSimilarity2DTransformFactory
 * \brief Object Factory implementation for GPUSimilarity2DTransform
 *
 * \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands
 *
 * \note This work was funded by the Netherlands Organisation for
 * Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
 *
 */
template <typename NDimensions>
class ITK_TEMPLATE_EXPORT GPUSimilarity2DTransformFactory2 : public GPUObjectFactoryBase<NDimensions>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(GPUSimilarity2DTransformFactory2);

  using Self = GPUSimilarity2DTransformFactory2;
  using Superclass = GPUObjectFactoryBase<NDimensions>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Return a descriptive string describing the factory. */
  const char *
  GetDescription() const
  {
    return "A Factory for GPUSimilarity2DTransform";
  }

  /** Method for class instantiation. */
  itkFactorylessNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(GPUSimilarity2DTransformFactory2, GPUObjectFactoryBase);

  /** Register one factory of this type. */
  static void
  RegisterOneFactory();

  /** Operator() to register override. */
  template <typename TType>
  void
  operator()()
  {
    this->RegisterOverride(typeid(Similarity2DTransform<TType>).name(),
                           typeid(GPUSimilarity2DTransform<TType>).name(),
                           "GPU Similarity2DTransform override",
                           true,
                           CreateObjectFunction<GPUSimilarity2DTransform<TType>>::New());
  }


protected:
  GPUSimilarity2DTransformFactory2();
  virtual ~GPUSimilarity2DTransformFactory2() {}

  /** Typedef for real type list. */
  using RealTypeList = typelist::MakeTypeList<float, double>::Type;

  /** Register methods for 2D. */
  virtual void
  Register2D();
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkGPUSimilarity2DTransformFactory.hxx"
#endif

#endif /* itkGPUSimilarity2DTransformFactory_h */
