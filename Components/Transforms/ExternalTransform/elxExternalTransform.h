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
#ifndef elxExternalTransform_h
#define elxExternalTransform_h

#include "itkAdvancedCombinationTransform.h"
#include "elxTransformBase.h"

#include <itkMacro.h>
#include <itkSmartPointer.h>

namespace elastix
{

/**
 * \class ExternalTransform
 *
 * This class is represents an external ITK transform.
 *
 * The parameters used in this class are:
 * \parameter Transform: Select this transform as follows:\n
 *    <tt>(Transform "ExternalTransform")</tt>
 * \parameter TransformFileName: \n
 *    example: <tt>(TransformFileName "Transform.h5")</tt> \n
 *    By default "" is assumed.
 *
 * \ingroup Transforms
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT ExternalTransform
  : public itk::AdvancedCombinationTransform<typename elx::TransformBase<TElastix>::CoordRepType,
                                             elx::TransformBase<TElastix>::FixedImageDimension>
  , public elx::TransformBase<TElastix>
{
public:
  ITK_DISALLOW_COPY_AND_ASSIGN(ExternalTransform);

  /** Standard ITK-stuff. */
  using Self = ExternalTransform;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  using ITKTransformType = itk::Transform<double, TElastix::FixedDimension, TElastix::MovingDimension>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ExternalTransform, itk::AdvancedCombinationTransform);

  /** Name of this class.
   * Use this name in the parameter file to select this specific transform. \n
   * example: <tt>(Transform "ExternalTransform")</tt>\n
   */
  elxClassNameMacro("ExternalTransform");

  ITKTransformType *
  GetITKTransform(void) override
  {
    std::string transformFileName;
    this->m_Configuration->ReadParameter(transformFileName, "TransformFileName", 0);

    if (transformFileName != m_TransformFileName)
    {
      m_ITKTransform = dynamic_cast<ITKTransformType *>(TransformIO::Read(transformFileName).GetPointer());
      m_TransformFileName = std::move(transformFileName);
    }
    return m_ITKTransform.GetPointer();
  }

private:
  /** Default-constructor. */
  ExternalTransform() = default;

  /** Destructor. */
  ~ExternalTransform() override = default;

  elxOverrideGetSelfMacro;

  /** Creates a map of the parameters specific for this (derived) transform type. */
  itk::ParameterFileParser::ParameterMapType
  CreateDerivedTransformParametersMap(void) const override;

  itk::SmartPointer<ITKTransformType> m_ITKTransform;

  std::string m_TransformFileName;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxExternalTransform.hxx"
#endif

#endif // end #ifndef elxExternalTransform_h
