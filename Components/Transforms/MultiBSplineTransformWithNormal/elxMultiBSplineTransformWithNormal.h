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
#ifndef elxMultiBSplineTransformWithNormal_h
#define elxMultiBSplineTransformWithNormal_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkAdvancedCombinationTransform.h"
#include "itkAdvancedBSplineDeformableTransform.h"
#include "itkMultiBSplineDeformableTransformWithNormal.h"

#include "itkGridScheduleComputer.h"
#include "itkUpsampleBSplineParametersFilter.h"

namespace elastix
{

/**
 * \class MultiBSplineTransformWithNormal
 * \brief A transform based on the itkMultiBSplineDeformableTransformWithNormal.
 *
 * This transform is a composition of B-spline transformations, allowing sliding motion between different labels.
 *
 * The parameters used in this class are:
 * \parameter Transform: Select this transform as follows:\n
 *    <tt>(%Transform "MultiBSplineTransformWithNormal")</tt>
 * \parameter BSplineTransformSplineOrder: choose a B-spline order 1,2, or 3. \n
 *    example: <tt>(BSplineTransformSplineOrder 3)</tt>\n
 *    Default value: 3 (cubic B-splines).
 * \parameter FinalGridSpacingInVoxels: the grid spacing of the B-spline transform for each dimension. \n
 *    example: <tt>(FinalGridSpacingInVoxels 8.0 8.0 8.0)</tt> \n
 *    If only one argument is given, that factor is used for each dimension. The spacing
 *    is not in millimeters, but in "voxel size units".
 *    The default is 16.0 in every dimension.
 * \parameter FinalGridSpacingInPhysicalUnits: the grid spacing of the B-spline transform for each dimension. \n
 *    example: <tt>(FinalGridSpacingInPhysicalUnits 8.0 8.0 8.0)</tt> \n
 *    If only one argument is given, that factor is used for each dimension. The spacing
 *    is specified in millimeters.
 *    If not specified, the FinalGridSpacingInVoxels is used, or the FinalGridSpacing,
 *    to compute a FinalGridSpacingInPhysicalUnits. If those are not specified, the default
 *    value for FinalGridSpacingInVoxels is used to compute a FinalGridSpacingInPhysicalUnits.
 *    If an affine transformation is provided as initial transformation, the control grid
 *    will be scaled to cover the fixed image domain in the space defined by the initial transformation.
 * \parameter GridSpacingSchedule: the grid spacing downsampling factors for the B-spline transform
 *    for each dimension and each resolution. \n
 *    example: <tt>(GridSpacingSchedule 4.0 4.0 2.0 2.0 1.0 1.0)</tt> \n
 *    Which is an example for a 2D image, using 3 resolutions. \n
 *    For convenience, you may also specify only one value for each resolution:\n
 *    example: <tt>(GridSpacingSchedule 4.0 2.0 1.0 )</tt> \n
 *    which is equivalent to the example above.
 *
 *
 * The transform parameters necessary for transformix, additionally defined by this class, are:
 * \transformparameter GridSize: stores the size of the B-spline grid. \n
 *    example: <tt>(GridSize 16 16 16)</tt>
 * \transformparameter GridIndex: stores the index of the B-spline grid. \n
 *    example: <tt>(GridIndex 0 0 0)</tt>
 * \transformparameter GridSpacing: stores the spacing of the B-spline grid. \n
 *    example: <tt>(GridSpacing 16.0 16.0 16.0)</tt>
 * \transformparameter GridOrigin: stores the origin of the B-spline grid. \n
 *    example: <tt>(GridOrigin 0.0 0.0 0.0)</tt>
 * \transformparameter GridDirection: stores the direction cosines of the B-spline grid. \n
 *    example: <tt>(GridDirection 1.0 0.0 0.0  0.0 1.0 0.0  0.0 0.0 0.1)</tt>
 * \transformparameter BSplineTransformSplineOrder: stores the B-spline order 1,2, or 3. \n
 *    example: <tt>(BSplineTransformSplineOrder 3)</tt>
 *    Default value: 3 (cubic B-splines).
 *
 * \todo It is unsure what happens when one of the image dimensions has length 1.
 *
 * \author Vivien Delmon
 *
 * \ingroup Transforms
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT MultiBSplineTransformWithNormal
  : public itk::AdvancedCombinationTransform<typename elx::TransformBase<TElastix>::CoordRepType,
                                             elx::TransformBase<TElastix>::FixedImageDimension>
  , public TransformBase<TElastix>
{
public:
  /** Standard ITK-stuff. */
  typedef MultiBSplineTransformWithNormal Self;
  typedef itk::AdvancedCombinationTransform<typename elx::TransformBase<TElastix>::CoordRepType,
                                            elx::TransformBase<TElastix>::FixedImageDimension>
                                        Superclass1;
  typedef elx::TransformBase<TElastix>  Superclass2;
  typedef itk::SmartPointer<Self>       Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(MultiBSplineTransformWithNormal, AdvancedCombinationTransform);

  /** Name of this class.
   * Use this name in the parameter file to select this specific transform. \n
   * example: <tt>(Transform "BSplineTransform")</tt>\n
   */
  elxClassNameMacro("MultiBSplineTransformWithNormal");

  /** Dimension of the fixed image. */
  itkStaticConstMacro(SpaceDimension, unsigned int, Superclass2::FixedImageDimension);

  /** The ITK-class that provides most of the functionality, and
   * that is set as the "CurrentTransform" in the CombinationTransform.
   */
  typedef itk::AdvancedBSplineDeformableTransformBase<typename elx::TransformBase<TElastix>::CoordRepType,
                                                      itkGetStaticConstMacro(SpaceDimension)>
                                                     BSplineTransformBaseType;
  typedef typename BSplineTransformBaseType::Pointer BSplineTransformBasePointer;

  /** Typedef for supported BSplineTransform types. */
  typedef itk::MultiBSplineDeformableTransformWithNormal<typename elx::TransformBase<TElastix>::CoordRepType,
                                                         itkGetStaticConstMacro(SpaceDimension),
                                                         1>
    MultiBSplineTransformWithNormalLinearType;
  typedef itk::MultiBSplineDeformableTransformWithNormal<typename elx::TransformBase<TElastix>::CoordRepType,
                                                         itkGetStaticConstMacro(SpaceDimension),
                                                         2>
    MultiBSplineTransformWithNormalQuadraticType;
  typedef itk::MultiBSplineDeformableTransformWithNormal<typename elx::TransformBase<TElastix>::CoordRepType,
                                                         itkGetStaticConstMacro(SpaceDimension),
                                                         3>
    MultiBSplineTransformWithNormalCubicType;

  /** Typedefs inherited from the superclass. */
  typedef typename Superclass1::ScalarType                ScalarType;
  typedef typename Superclass1::ParametersType            ParametersType;
  typedef typename Superclass1::NumberOfParametersType    NumberOfParametersType;
  typedef typename Superclass1::JacobianType              JacobianType;
  typedef typename Superclass1::InputVectorType           InputVectorType;
  typedef typename Superclass1::OutputVectorType          OutputVectorType;
  typedef typename Superclass1::InputCovariantVectorType  InputCovariantVectorType;
  typedef typename Superclass1::OutputCovariantVectorType OutputCovariantVectorType;
  typedef typename Superclass1::InputVnlVectorType        InputVnlVectorType;
  typedef typename Superclass1::OutputVnlVectorType       OutputVnlVectorType;
  typedef typename Superclass1::InputPointType            InputPointType;
  typedef typename Superclass1::OutputPointType           OutputPointType;

  /** Typedef's specific for the BSplineTransform. */
  typedef typename BSplineTransformBaseType::PixelType               PixelType;
  typedef typename BSplineTransformBaseType::ImageType               ImageType;
  typedef typename BSplineTransformBaseType::ImagePointer            ImagePointer;
  typedef typename BSplineTransformBaseType::RegionType              RegionType;
  typedef typename BSplineTransformBaseType::IndexType               IndexType;
  typedef typename BSplineTransformBaseType::SizeType                SizeType;
  typedef typename BSplineTransformBaseType::SpacingType             SpacingType;
  typedef typename BSplineTransformBaseType::OriginType              OriginType;
  typedef typename BSplineTransformBaseType::DirectionType           DirectionType;
  typedef typename BSplineTransformBaseType::ContinuousIndexType     ContinuousIndexType;
  typedef typename BSplineTransformBaseType::ParameterIndexArrayType ParameterIndexArrayType;

  /** Typedef's from TransformBase. */
  typedef typename Superclass2::ElastixType              ElastixType;
  typedef typename Superclass2::ElastixPointer           ElastixPointer;
  typedef typename Superclass2::ParameterMapType         ParameterMapType;
  typedef typename Superclass2::ConfigurationType        ConfigurationType;
  typedef typename Superclass2::ConfigurationPointer     ConfigurationPointer;
  typedef typename Superclass2::RegistrationType         RegistrationType;
  typedef typename Superclass2::RegistrationPointer      RegistrationPointer;
  typedef typename Superclass2::CoordRepType             CoordRepType;
  typedef typename Superclass2::FixedImageType           FixedImageType;
  typedef typename Superclass2::MovingImageType          MovingImageType;
  typedef typename Superclass2::ITKBaseType              ITKBaseType;
  typedef typename Superclass2::CombinationTransformType CombinationTransformType;

  /** Typedef's for the GridScheduleComputer and the UpsampleBSplineParametersFilter. */
  typedef itk::GridScheduleComputer<CoordRepType, SpaceDimension>         GridScheduleComputerType;
  typedef typename GridScheduleComputerType::Pointer                      GridScheduleComputerPointer;
  typedef typename GridScheduleComputerType ::VectorGridSpacingFactorType GridScheduleType;
  typedef itk::UpsampleBSplineParametersFilter<ParametersType, ImageType> GridUpsamplerType;
  typedef typename GridUpsamplerType::Pointer                             GridUpsamplerPointer;

  /** Typdef's for the Image of Labels */
  typedef itk::Image<unsigned char, itkGetStaticConstMacro(SpaceDimension)> ImageLabelType;
  typedef typename ImageLabelType::Pointer                                  ImageLabelPointer;

  /** Execute stuff before anything else is done:
   * \li Initialize the right BSplineTransform.
   * \li Initialize the right grid schedule computer.
   */
  int
  BeforeAll(void) override;

  /** Execute stuff before the actual registration:
   * \li Create an initial B-spline grid.
   * \li Create initial registration parameters.
   * \li PrecomputeGridInformation
   * Initially, the transform is set to use a 1x1x1 grid, with deformation (0,0,0).
   * In the method BeforeEachResolution() this will be replaced by the right grid size.
   * This seems not logical, but it is required, since the registration
   * class checks if the number of parameters in the transform is equal to
   * the number of parameters in the registration class. This check is done
   * before calling the BeforeEachResolution() methods.
   */
  void
  BeforeRegistration(void) override;

  /** Execute stuff before each new pyramid resolution:
   * \li In the first resolution call InitializeTransform().
   * \li In next resolutions upsample the B-spline grid if necessary (so, call IncreaseScale())
   */
  void
  BeforeEachResolution(void) override;

  /** Method to set the initial BSpline grid and initialize the parameters (to 0).
   * \li Define the initial grid region, origin and spacing, using the precomputed grid information.
   * \li Set the initial parameters to zero and set then as InitialParametersOfNextLevel in the registration object.
   * Called by BeforeEachResolution().
   */
  virtual void
  InitializeTransform(void);

  /** Method to increase the density of the BSpline grid.
   * \li Determine the new B-spline coefficients that describe the current deformation field.
   * \li Set these coefficients as InitialParametersOfNextLevel in the registration object.
   * Called by BeforeEachResolution().
   */
  virtual void
  IncreaseScale(void);

  /** Function to read transform-parameters from a file. */
  void
  ReadFromFile(void) override;

  /** Set the scales of the edge B-spline coefficients to zero. */
  virtual void
  SetOptimizerScales(const unsigned int edgeWidth);

protected:
  /** The constructor. */
  MultiBSplineTransformWithNormal();

  /** The destructor. */
  ~MultiBSplineTransformWithNormal() override = default;

  /** Read user-specified gridspacing and call the itkGridScheduleComputer. */
  virtual void
  PreComputeGridInformation(void);

private:
  elxOverrideGetSelfMacro;

  /** Creates a map of the parameters specific for this (derived) transform type. */
  ParameterMapType
  CreateDerivedTransformParametersMap(void) const override;

  /** The deleted copy constructor. */
  MultiBSplineTransformWithNormal(const Self &) = delete;
  /** The deleted assignment operator. */
  void
  operator=(const Self &) = delete;

  /** Private variables. */
  typename MultiBSplineTransformWithNormalCubicType::Pointer m_MultiBSplineTransformWithNormal;
  GridScheduleComputerPointer                                m_GridScheduleComputer;
  GridUpsamplerPointer                                       m_GridUpsampler;
  ImageLabelPointer                                          m_Labels;
  std::string                                                m_LabelsPath;

  /** Variable to remember order of MultiBSplineTransformWithNormal. */
  unsigned int m_SplineOrder;

  /** Initialize the right BSplineTransfrom based on the spline order and periodicity. */
  unsigned int
  InitializeBSplineTransform();
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxMultiBSplineTransformWithNormal.hxx"
#endif

#endif // end #ifndef elxMultiBSplineTransformWithNormal_h
