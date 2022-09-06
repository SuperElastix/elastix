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
#ifndef elxTransformixMain_h
#define elxTransformixMain_h

#include "elxElastixMain.h"

namespace elastix
{
/**
 * \class TransformixMain
 * \brief A class with all functionality to configure transformix.
 *
 * The TransformixMain class inherits from ElastixMain. We overwrite the Run()
 * -function. In the new Run() the Run()-function from the
 * ElastixTemplate-class is not called (as in elxElastixMain.cxx),
 * because this time we don't want to start a registration, but
 * just apply a transformation to an input image.
 *
 * \ingroup Kernel
 */

class TransformixMain : public ElastixMain
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(TransformixMain);

  /** Standard itk. */
  using Self = TransformixMain;
  using Superclass = ElastixMain;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(TransformixMain, ElastixMain);

  /** Typedef's from Superclass. */

  /** typedef's from itk base Object. */
  using Superclass::ObjectPointer;
  using Superclass::DataObjectPointer;

  /** Elastix components. */
  using Superclass::ArgumentMapType;
  using Superclass::ObjectContainerType;
  using Superclass::DataObjectContainerType;
  using Superclass::ObjectContainerPointer;
  using Superclass::DataObjectContainerPointer;

  /** Typedefs for the database that holds pointers to New() functions.
   * Those functions are used to instantiate components, such as the metric etc.
   */
  using Superclass::ComponentDatabasePointer;
  using Superclass::PtrToCreator;
  using Superclass::ComponentDescriptionType;
  using Superclass::PixelTypeDescriptionType;
  using Superclass::ImageDimensionType;
  using Superclass::DBIndexType;

  /** Typedef that is used in the elastix dll version. */
  using Superclass::ParameterMapType;

  /** Overwrite Run() from base-class. */
  int
  Run() override;

  /** Overwrite Run( argmap ) from superclass. Simply calls the superclass. */
  int
  Run(const ArgumentMapType & argmap) override;

  int
  Run(const ArgumentMapType & argmap, const ParameterMapType & inputMap) override;

  /** Run version for using transformix as library. */
  virtual int
  Run(const ArgumentMapType & argmap, const std::vector<ParameterMapType> & inputMaps);

  /** Get and Set input- and outputImage. */
  virtual void
  SetInputImageContainer(DataObjectContainerType * inputImageContainer);

protected:
  TransformixMain() = default;
  ~TransformixMain() override;

  /** InitDBIndex sets m_DBIndex to the value obtained
   * from the ComponentDatabase.
   */
  int
  InitDBIndex() override;
};

} // end namespace elastix

#endif // end #ifndef elxTransformixMain_h
