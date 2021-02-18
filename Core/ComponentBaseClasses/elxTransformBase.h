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
#ifndef elxTransformBase_h
#define elxTransformBase_h

/** Needed for the macros */
#include "elxMacro.h"

#include "elxBaseComponentSE.h"
#include "elxElastixBase.h"
#include "itkAdvancedTransform.h"
#include "itkAdvancedCombinationTransform.h"
#include "elxComponentDatabase.h"
#include "elxProgressCommand.h"

// ITK header files:
#include <itkImage.h>
#include <itkOptimizerParameters.h>

#include <memory> // For unique_ptr.

namespace elastix
{
// using namespace itk; //Not here, because a TransformBase class was added to ITK...

/**
 * \class TransformBase
 * \brief This class is the elastix base class for all Transforms.
 *
 * This class contains the common functionality for all Transforms.
 *
 * The parameters used in this class are:
 * \parameter HowToCombineTransforms: Indicates how to use the initial transform\n
 *   (given by the command-line argument -t0, or, if using multiple parameter files,
 *   by the result of registration using the previous parameter file). Possible options
 *   are "Add" and "Compose".\n
 *   "Add" combines the initial transform \f$T_0\f$ and the current
 *   transform \f$T_1\f$ (which is currently optimized) by
 *   addition: \f$T(x) = T_0(x) + T_1(x)\f$;\n
 *   "Compose" by composition: \f$T(x) = T_1 ( T_0(x) )\f$.\n
 *   example: <tt>(HowToCombineTransforms "Add")</tt>\n
 *   Default: "Add".
 *
 * \transformparameter UseDirectionCosines: Controls whether to use or ignore the
 * direction cosines (world matrix, transform matrix) set in the images.
 * Voxel spacing and image origin are always taken into account, regardless
 * the setting of this parameter.\n
 *    example: <tt>(UseDirectionCosines "true")</tt>\n
 * Default: true. Setting it to false means that you choose to ignore important
 * information from the image, which relates voxel coordinates to world coordinates.
 * Ignoring it may easily lead to left/right swaps for example, which could
 * skrew up a (medical) analysis.
 * \transformparameter HowToCombineTransforms: Indicates how to use the initial transform
 *   (given by the command-line argument -t0, or, if using multiple parameter files,
 *   by the result of registration using the previous parameter file). Possible options
 *   are "Add" and "Compose".\n
 *   "Add" combines the initial transform \f$T_0\f$ and the current
 *   transform \f$T_1\f$ (which is currently optimized) by
 *   addition: \f$T(x) = T_0(x) + T_1(x)\f$;\n
 *   "Compose" by composition: \f$T(x) = T_1 ( T_0(x) )\f$.\n
 *   example: <tt>(HowToCombineTransforms "Add")</tt>\n
 *   Default: "Compose".
 * \transformparameter Size: The size (number of voxels in each dimension) of the fixed image
 * that was used during registration, and which is used for resampling the deformed moving image.\n
 * example: <tt>(Size 100 90 90)</tt>\n
 * Mandatory parameter.
 * \transformparameter Index: The starting index of the fixed image region
 * that was used during registration, and which is used for resampling the deformed moving image.\n
 * example: <tt>(Index 0 0 0)</tt>\n
 * Currently always zero.
 * \transformparameter Spacing: The voxel spacing of the fixed image
 * that was used during registration, and which is used for resampling the deformed moving image.\n
 * example: <tt>(Spacing 1.0 1.0 1.0)</tt>\n
 * Default: 1.0 for each dimension.
 * \transformparameter Origin: The origin (location of the first voxel in world coordinate) of the fixed image
 * that was used during registration, and which is used for resampling the deformed moving image.\n
 * example: <tt>(Origin 5.0 10.0 11.0)</tt>\n
 * Default: 0.0 for each dimension.
 * \transformparameter Direction: The direction cosines matrix of the fixed image
 * that was used during registration, and which is used for resampling the deformed moving image
 * if the UseDirectionCosines parameter is set to "true".\n
 * example: <tt>(Direction -1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.1)</tt>\n
 * Default: identity matrix. Elements are sorted as follows: [ d11 d21 d31 d12 d22 d32 d13 d23 d33] (in 3D).
 * \transformparameter TransformParameters: the transform parameter vector that defines the transformation.\n
 * example <tt>(TransformParameters 0.03 1.0 0.2 ...)</tt>\n
 * The number of entries is stored the NumberOfParameters entry.
 * \transformparameter NumberOfParameters: the length of the transform parameter vector.\n
 * example <tt>(NumberOfParameters 722)</tt>\n
 * \transformparameter InitialTransformParametersFileName: The location/name of an initial
 * transform that will be loaded when loading the current transform parameter file. Note
 * that transform parameter file can also contain an initial transform. Recursively all
 * transforms are thus automatically loaded when loading the last transform parameter file.\n
 * example <tt>(InitialTransformParametersFileName "./res/TransformParameters.0.txt")</tt>\n
 * The location is relative to the path from where elastix/transformix is started!\n
 * Default: "NoInitialTransform", which (obviously) means that there is no initial transform
 * to be loaded.
 *
 * The command line arguments used by this class are:
 * \commandlinearg -t0: optional argument for elastix for specifying an initial transform
 *    parameter file. \n
 *    example: <tt>-t0 TransformParameters.txt</tt> \n
 * \commandlinearg -def: optional argument for transformix for specifying a set of points
 *    that have to be transformed.\n
 *    example: <tt>-def inputPoints.txt</tt> \n
 *    The inputPoints.txt file should be structured: first line should be "index" or
 *    "point", depending if the user supplies voxel indices or real world coordinates.
 *    The second line should be the number of points that should be transformed. The
 *    third and following lines give the indices or points.\n
 *    It is also possible to deform all points, thereby generating a deformation field
 *    image. This is done by:\n
 *    example: <tt>-def all</tt> \n
 *
 * \ingroup Transforms
 * \ingroup ComponentBaseClasses
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT TransformBase : public BaseComponentSE<TElastix>
{
public:
  ITK_DISALLOW_COPY_AND_ASSIGN(TransformBase);

  /** Standard ITK stuff. */
  typedef TransformBase             Self;
  typedef BaseComponentSE<TElastix> Superclass;

  /** Run-time type information (and related methods). */
  itkTypeMacro(TransformBase, BaseComponentSE);

  /** Typedef from Superclass. */
  typedef typename Superclass::RegistrationType RegistrationType;

  typedef Configuration ::CommandLineArgumentMapType CommandLineArgumentMapType;
  typedef Configuration ::CommandLineEntryType       CommandLineEntryType;

  /** Elastix typedef's. */
  typedef ElastixBase::CoordRepType          CoordRepType;
  typedef typename TElastix::FixedImageType  FixedImageType;
  typedef typename TElastix::MovingImageType MovingImageType;

  /** Typedef's from ComponentDatabase. */
  typedef ComponentDatabase                               ComponentDatabaseType;
  typedef ComponentDatabaseType::ComponentDescriptionType ComponentDescriptionType;
  typedef ComponentDatabase::PtrToCreator                 PtrToCreator;

  /** Typedef for the ProgressCommand. */
  typedef elx::ProgressCommand ProgressCommandType;

  /** Get the dimension of the fixed image. */
  itkStaticConstMacro(FixedImageDimension, unsigned int, FixedImageType::ImageDimension);

  /** Get the dimension of the moving image. */
  itkStaticConstMacro(MovingImageDimension, unsigned int, MovingImageType::ImageDimension);

  /** Other typedef's. */
  typedef itk::Object ObjectType;
  typedef itk::AdvancedCombinationTransform<CoordRepType, itkGetStaticConstMacro(FixedImageDimension)>
                                                                  CombinationTransformType;
  typedef CombinationTransformType                                ITKBaseType;
  typedef typename CombinationTransformType::InitialTransformType InitialTransformType;

  /** Typedef's for parameters. */
  using ValueType = double;
  using ParametersType = itk::OptimizerParameters<ValueType>;

  /** Typedef's for TransformPoint. */
  typedef typename ITKBaseType::InputPointType  InputPointType;
  typedef typename ITKBaseType::OutputPointType OutputPointType;

  /** Typedef's for TransformPointsAllPoints. */
  typedef itk::Vector<float, FixedImageDimension>          VectorPixelType;
  typedef itk::Image<VectorPixelType, FixedImageDimension> DeformationFieldImageType;

  /** Typedefs needed for AutomaticScalesEstimation function */
  typedef typename RegistrationType::ITKBaseType      ITKRegistrationType;
  typedef typename ITKRegistrationType::OptimizerType OptimizerType;
  typedef typename OptimizerType::ScalesType          ScalesType;

  /** Typedef that is used in the elastix dll version. */
  typedef typename TElastix::ParameterMapType ParameterMapType;

  /** Retrieves this object as ITKBaseType. */
  ITKBaseType *
  GetAsITKBaseType(void)
  {
    return &(this->GetSelf());
  }


  /** Retrieves this object as ITKBaseType, to use in const functions. */
  const ITKBaseType *
  GetAsITKBaseType(void) const
  {
    return &(this->GetSelf());
  }

  /** Execute stuff before the actual transformation:
   * \li Check the appearance of inputpoints to be transformed.
   */
  int
  BeforeAllTransformix(void);

  /** Set the initial transform. */
  void
  SetInitialTransform(InitialTransformType * _arg);

  /** Set the TransformParametersFileName. */
  void
  SetTransformParametersFileName(const char * filename);

  /** Function to read transform-parameters from a file. */
  virtual void
  ReadFromFile(void);

  /** Function to create transform-parameters map. */
  void
  CreateTransformParametersMap(const ParametersType & param, ParameterMapType & parameterMap) const;

  /** Function to write transform-parameters to a file. */
  void
  WriteToFile(xl::xoutsimple & transformationParameterInfo, const ParametersType & param) const;

  /** Macro for reading and writing the transform parameters in WriteToFile or not. */
  void
  SetReadWriteTransformParameters(const bool _arg);

  /** Function to read the initial transform parameters from a file. */
  void
  ReadInitialTransformFromFile(const char * transformParameterFileName);

  /** Function to read the initial transform parameters from the internally stored
   * configuration object.
   */
  void
  ReadInitialTransformFromVector(const size_t index);

  /** Function to transform coordinates from fixed to moving image. */
  void
  TransformPoints(void) const;

  /** Function to compute the determinant of the spatial Jacobian. */
  void
  ComputeDeterminantOfSpatialJacobian(void) const;

  /** Function to compute the determinant of the spatial Jacobian. */
  void
  ComputeSpatialJacobian(void) const;

  /** Makes sure that the final parameters from the registration components
   * are copied, set, and stored.
   */
  void
  SetFinalParameters(void);

protected:
  /** The default-constructor. */
  TransformBase() = default;
  /** The destructor. */
  ~TransformBase() override = default;

  /** Estimate a scales vector
   * AutomaticScalesEstimation works like this:
   * \li N=10000 points are sampled on a uniform grid on the fixed image.
   * \li Jacobians dT/dmu are computed
   * \li Scales_i = 1/N sum_x || dT / dmu_i ||^2
   */
  void
  AutomaticScalesEstimation(ScalesType & scales) const;

  /** Estimate a scales vector for a stack transform (elxTranslationStackTransform,
   * elxAffineStackTransform, ...) Instead of sampling along the n dimensions of the
   * fixed image, it samples along n-1 dimensions. Then
   * \li N=10000 points are sampled.
   * \li Jacobians dT/dmu are computed
   * \li Scales_i = 1/N sum_x || dT / dmu_i ||^2
   */
  void
  AutomaticScalesEstimationStackTransform(const unsigned int & numSubTransforms, ScalesType & scales) const;

private:
  elxDeclarePureVirtualGetSelfMacro(ITKBaseType);

  /** Function to read the initial transform parameters from the specified configuration object.
   */
  void
  ReadInitialTransformFromConfiguration(const Configuration::Pointer);

  /** Execute stuff before everything else:
   * \li Check the appearance of an initial transform.
   */
  int
  BeforeAllBase(void) override;

  /** Execute stuff before the actual registration:
   * \li Set the initial transform and how to group transforms.
   */
  void
  BeforeRegistrationBase(void) override;

  /** Execute stuff after the registration:
   * \li Get and set the final parameters for the resampler.
   */
  void
  AfterRegistrationBase(void) override;

  /** Get the initial transform. */
  const InitialTransformType *
  GetInitialTransform(void) const;

  /** Get the TransformParametersFileName. */
  itkGetStringMacro(TransformParametersFileName);

  /** Function to transform coordinates from fixed to moving image. */
  void
  TransformPointsSomePoints(const std::string & filename) const;

  /** Function to transform coordinates from fixed to moving image, given as VTK file. */
  void
  TransformPointsSomePointsVTK(const std::string & filename) const;

  /** Deprecation note: The plan is to split all Compute* and TransformPoints* functions
   *  into Generate* and Write* functions, since that would facilitate a proper library
   *  interface. To keep everything functional during the transition period we need to
   *  keep the orignal Compute* and TransformPoints* functions and let them just call
   *  Generate* and Write*. These functions should be considered marked deprecated.
   */

  /** Function to transform all coordinates from fixed to moving image. */
  typename DeformationFieldImageType::Pointer
  GenerateDeformationFieldImage(void) const;

  void WriteDeformationFieldImage(typename DeformationFieldImageType::Pointer) const;

  /** Legacy function that calls GenerateDeformationFieldImage and WriteDeformationFieldImage. */
  void
  TransformPointsAllPoints(void) const;

  std::string
  GetInitialTransformParametersFileName(void) const
  {
    if (!this->GetInitialTransform())
    {
      return "NoInitialTransform";
    }

    const Self * t0 = dynamic_cast<const Self *>(this->GetInitialTransform());
    return t0->GetTransformParametersFileName();
  }

  virtual ParameterMapType
  CreateDerivedTransformParametersMap(void) const = 0;

  /** Allows a derived transform class to write its data to file, by overriding this member function. */
  virtual void
  WriteDerivedTransformDataToFile(void) const
  {}

  /** Member variables. */
  std::unique_ptr<ParametersType> m_TransformParametersPointer{};
  std::string                     m_TransformParametersFileName;
  ParametersType                  m_FinalParameters;

  /** Boolean to decide whether or not the transform parameters are written. */
  bool m_ReadWriteTransformParameters{ true };

  /** Boolean to decide whether or not the transform parameters are written in binary format. */
  bool m_UseBinaryFormatForTransformationParameters{};
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxTransformBase.hxx"
#endif

#endif // end #ifndef elxTransformBase_h
