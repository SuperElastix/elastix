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
#ifndef itkGPUTranslationTransform_h
#define itkGPUTranslationTransform_h

#include "itkTranslationTransform.h"
#include "itkGPUTranslationTransformBase.h"

namespace itk
{
/** \class GPUTranslationTransform
 * \brief GPU version of TranslationTransform.
 *
 * \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands
 *
 * \note This work was funded by the Netherlands Organisation for
 * Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
 *
 * \ingroup GPUCommon
 */
template <typename TScalarType = float,
          unsigned int NDimensions = 3,
          typename TParentTransform = TranslationTransform<TScalarType, NDimensions>>
class GPUTranslationTransform
  : public TParentTransform
  , public GPUTranslationTransformBase<TScalarType, NDimensions>
{
public:
  /** Standard class typedefs. */
  typedef GPUTranslationTransform                               Self;
  typedef TParentTransform                                      CPUSuperclass;
  typedef GPUTranslationTransformBase<TScalarType, NDimensions> GPUSuperclass;
  typedef SmartPointer<Self>                                    Pointer;
  typedef SmartPointer<const Self>                              ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(GPUTranslationTransform, CPUSuperclass);

  /** Typedefs from GPUSuperclass. */
  typedef typename GPUSuperclass::CPUOutputVectorType CPUOutputVectorType;

  /** This method returns the CPU value of the offset of the TranslationTransform. */
  virtual const CPUOutputVectorType &
  GetCPUOffset(void) const
  {
    return this->GetOffset();
  }

protected:
  GPUTranslationTransform() {}
  virtual ~GPUTranslationTransform() {}

private:
  GPUTranslationTransform(const Self & other) = delete;
  const Self &
  operator=(const Self &) = delete;
};

} // end namespace itk

#endif /* itkGPUTranslationTransform_h */
