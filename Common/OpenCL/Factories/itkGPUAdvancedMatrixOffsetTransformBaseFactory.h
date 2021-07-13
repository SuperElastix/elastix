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
#ifndef itkGPUAdvancedMatrixOffsetTransformBaseFactory_h
#define itkGPUAdvancedMatrixOffsetTransformBaseFactory_h

#include "itkGPUObjectFactoryBase.h"
#include "itkGPUAdvancedMatrixOffsetTransformBase.h"

namespace itk
{
/** \class GPUAdvancedMatrixOffsetTransformBaseFactory
 * \brief Object Factory implementation for GPUAdvancedMatrixOffsetTransformBase
 *
 * \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands
 *
 * \note This work was funded by the Netherlands Organisation for
 * Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
 *
 */
template <typename NDimensions>
class ITK_TEMPLATE_EXPORT GPUAdvancedMatrixOffsetTransformBaseFactory2 : public GPUObjectFactoryBase<NDimensions>
{
public:
  typedef GPUAdvancedMatrixOffsetTransformBaseFactory2 Self;
  typedef GPUObjectFactoryBase<NDimensions>            Superclass;
  typedef SmartPointer<Self>                           Pointer;
  typedef SmartPointer<const Self>                     ConstPointer;

  /** Return a descriptive string describing the factory. */
  const char *
  GetDescription() const
  {
    return "A Factory for GPUAdvancedMatrixOffsetTransformBase";
  }

  /** Method for class instantiation. */
  itkFactorylessNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(GPUAdvancedMatrixOffsetTransformBaseFactory2, GPUObjectFactoryBase);

  /** Register one factory of this type. */
  static void
  RegisterOneFactory();

  /** Operator() to register override. */
  template <typename TType, unsigned int VImageDimension>
  void
  operator()(void)
  {
    this->RegisterOverride(typeid(AdvancedMatrixOffsetTransformBase<TType, VImageDimension>).name(),
                           typeid(GPUAdvancedMatrixOffsetTransformBase<TType, VImageDimension>).name(),
                           "GPU AdvancedMatrixOffsetTransformBase override",
                           true,
                           CreateObjectFunction<GPUAdvancedMatrixOffsetTransformBase<TType, VImageDimension>>::New());
  }


protected:
  GPUAdvancedMatrixOffsetTransformBaseFactory2();
  virtual ~GPUAdvancedMatrixOffsetTransformBaseFactory2() {}

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
  GPUAdvancedMatrixOffsetTransformBaseFactory2(const Self &) = delete;
  void
  operator=(const Self &) = delete;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkGPUAdvancedMatrixOffsetTransformBaseFactory.hxx"
#endif

#endif /* itkGPUAdvancedMatrixOffsetTransformBaseFactory_h */
