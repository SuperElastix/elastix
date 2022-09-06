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
#ifndef elxFullSearchOptimizer_h
#define elxFullSearchOptimizer_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkFullSearchOptimizer.h"
#include <map>

#include "itkNDImageBase.h"

namespace elastix
{

/**
 * \class FullSearch
 * \brief An optimizer based on the itk::FullSearchOptimizer.
 *
 * Optimizer that scans a subspace of the parameter space
 * and searches for the best parameters.
 *
 * The results are written to the output-directory as an image
 * OptimizationSurface.\<elastixlevel\>.R\<resolution\>.mhd",
 * which is an N-dimensional float image, where N is the
 * dimension of the search space.
 *
 * The parameters used in this class are:
 * \parameter Optimizer: Select this optimizer as follows:\n
 *    <tt>(Optimizer "FullSearch")</tt>
 * \parameter FullSearchSpace\<r\>: Defines for resolution r a range of parameters to scan.\n
 *   Full syntax: (FullSearchSpace\<r\> \<parameter_name\> \<parameter_nr\> \<min\> \<max\> \<stepsize\> [...] ) \n
 *   example: <tt>(FullSearchSpace0 "translation_x" 2 -4.0 3.0 1.0 "rotation_y" 3 -1.0 1.0 0.5)</tt> \n
 *   This varies the second transform parameter in the range [-4.0 3.0] with steps of 1.0
 *   and the third parameter in the range [-1.0 1.0] with steps of 0.5. The names are used
 *   as column headers in the screen output.
 *
 * \ingroup Optimizers
 * \sa FullSearchOptimizer
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT FullSearch
  : public itk::FullSearchOptimizer
  , public OptimizerBase<TElastix>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(FullSearch);

  /** Standard ITK.*/
  using Self = FullSearch;
  using Superclass1 = itk::FullSearchOptimizer;
  using Superclass2 = OptimizerBase<TElastix>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(FullSearch, itk::FullSearchOptimizer);

  /** Name of this class.
   * Use this name in the parameter file to select this specific optimizer. \n
   * example: <tt>(Optimizer "FullSearch")</tt>\n
   */
  elxClassNameMacro("FullSearch");

  /** Typedef's inherited from Superclass1.*/
  using Superclass1::CostFunctionType;
  using Superclass1::CostFunctionPointer;
  using Superclass1::ParametersType;
  using Superclass1::MeasureType;
  using Superclass1::ParameterValueType;
  using Superclass1::RangeValueType;
  using Superclass1::RangeType;
  using Superclass1::SearchSpaceType;
  using Superclass1::SearchSpacePointer;
  using Superclass1::SearchSpaceIteratorType;
  using Superclass1::SearchSpacePointType;
  using Superclass1::SearchSpaceIndexType;
  using Superclass1::SearchSpaceSizeType;

  /** Typedef's inherited from Elastix.*/
  using typename Superclass2::ElastixType;
  using typename Superclass2::RegistrationType;
  using ITKBaseType = typename Superclass2::ITKBaseType;

  /** To store the results of the full search */
  using NDImageType = itk::NDImageBase<float>;
  using NDImagePointer = typename NDImageType::Pointer;

  /** To store the names of the search space dimensions */
  using DimensionNameMapType = std::map<unsigned int, std::string>;
  using NameIteratorType = typename DimensionNameMapType::const_iterator;

  /** Methods that have to be present everywhere.*/
  void
  BeforeRegistration() override;

  void
  BeforeEachResolution() override;

  void
  AfterEachResolution() override;

  void
  AfterEachIteration() override;

  void
  AfterRegistration() override;

  /** \todo BeforeAll, checking parameters. */

  /** Get a pointer to the image containing the optimization surface. */
  itkGetModifiableObjectMacro(OptimizationSurface, NDImageType);

protected:
  FullSearch();
  ~FullSearch() override = default;

  NDImagePointer m_OptimizationSurface;

  DimensionNameMapType m_SearchSpaceDimensionNames;

  /** Checks if an error generated while reading the search space
   * ranges from the parameter file is a real error. Prints some
   * error message if so.
   */
  // virtual int CheckSearchSpaceRangeDefinition(const std::string & fullFieldName,
  //  int errorcode, unsigned int entry_nr);
  virtual bool
  CheckSearchSpaceRangeDefinition(const std::string & fullFieldName,
                                  const bool          found,
                                  const unsigned int  entry_nr) const;

private:
  elxOverrideGetSelfMacro;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxFullSearchOptimizer.hxx"
#endif

#endif // end #ifndef elxFullSearchOptimizer_h
