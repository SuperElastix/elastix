/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __elxTransformBase_h
#define __elxTransformBase_h

/** Needed for the macros */
#include "elxMacro.h"

#include "elxBaseComponentSE.h"
#include "itkAdvancedTransform.h"
#include "itkAdvancedCombinationTransform.h"
#include "elxComponentDatabase.h"
#include "elxProgressCommand.h"

#include <fstream>
#include <iomanip>

namespace elastix
{
//using namespace itk; //Not here, because a TransformBase class was added to ITK...

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
 * Default: false. Recommended: true. The default value is false for
 * backward compatibility reasons. This parameter was introduced in
 * elastix 4.3. Setting it to false means that you choose to ignore important
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

template< class TElastix >
class TransformBase :
  public BaseComponentSE< TElastix >
{
public:

  /** Standard ITK stuff. */
  typedef TransformBase               Self;
  typedef BaseComponentSE< TElastix > Superclass;

  /** Run-time type information (and related methods). */
  itkTypeMacro( TransformBase, BaseComponentSE );

  /** Typedef's from Superclass. */
  typedef typename Superclass::ElastixType          ElastixType;
  typedef typename Superclass::ElastixPointer       ElastixPointer;
  typedef typename Superclass::ConfigurationType    ConfigurationType;
  typedef typename Superclass::ConfigurationPointer ConfigurationPointer;
  typedef typename ConfigurationType
    ::CommandLineArgumentMapType CommandLineArgumentMapType;
  typedef typename ConfigurationType
    ::CommandLineEntryType CommandLineEntryType;
  typedef typename Superclass::RegistrationType    RegistrationType;
  typedef typename Superclass::RegistrationPointer RegistrationPointer;

  /** Elastix typedef's. */
  typedef typename ElastixType::CoordRepType    CoordRepType;
  typedef typename ElastixType::FixedImageType  FixedImageType;
  typedef typename ElastixType::MovingImageType MovingImageType;

  /** Typedef's from ComponentDatabase. */
  typedef ComponentDatabase                               ComponentDatabaseType;
  typedef ComponentDatabaseType::ComponentDescriptionType ComponentDescriptionType;
  typedef ComponentDatabase::PtrToCreator                 PtrToCreator;

  /** Typedef for the ProgressCommand. */
  typedef elx::ProgressCommand ProgressCommandType;

  /** Get the dimension of the fixed image. */
  itkStaticConstMacro( FixedImageDimension,
    unsigned int, FixedImageType::ImageDimension );

  /** Get the dimension of the moving image. */
  itkStaticConstMacro( MovingImageDimension,
    unsigned int, MovingImageType::ImageDimension );

  /** Other typedef's. */
  typedef itk::Object ObjectType;
  typedef itk::AdvancedTransform<
    CoordRepType,
    itkGetStaticConstMacro( FixedImageDimension ),
    itkGetStaticConstMacro( MovingImageDimension ) >  ITKBaseType;
  typedef itk::AdvancedCombinationTransform< CoordRepType,
    itkGetStaticConstMacro( FixedImageDimension ) >   CombinationTransformType;
  typedef typename
    CombinationTransformType::InitialTransformType InitialTransformType;

  /** Typedef's from Transform. */
  typedef typename ITKBaseType::ParametersType ParametersType;
  typedef typename ParametersType::ValueType   ValueType;

  /** Typedef's for TransformPoint. */
  typedef typename ITKBaseType::InputPointType  InputPointType;
  typedef typename ITKBaseType::OutputPointType OutputPointType;

  /** Typedefs needed for AutomaticScalesEstimation function */
  typedef typename RegistrationType::ITKBaseType      ITKRegistrationType;
  typedef typename ITKRegistrationType::OptimizerType OptimizerType;
  typedef typename OptimizerType::ScalesType          ScalesType;

  /** Typedef that is used in the elastix dll version. */
  typedef typename ElastixType::ParameterMapType ParameterMapType;

  /** Cast to ITKBaseType. */
  virtual ITKBaseType * GetAsITKBaseType( void )
  {
    return dynamic_cast< ITKBaseType * >( this );
  }


  /** Cast to ITKBaseType, to use in const functions. */
  virtual const ITKBaseType * GetAsITKBaseType( void ) const
  {
    return dynamic_cast< const ITKBaseType * >( this );
  }


  virtual const CombinationTransformType * GetAsCombinationTransform( void ) const
  {
    return dynamic_cast< const CombinationTransformType * >( this );
  }


  virtual CombinationTransformType * GetAsCombinationTransform( void )
  {
    return dynamic_cast< CombinationTransformType * >( this );
  }


  /** Execute stuff before everything else:
   * \li Check the appearance of an initial transform.
   */
  virtual int BeforeAllBase( void );

  /** Execute stuff before the actual transformation:
   * \li Check the appearance of inputpoints to be transformed.
   */
  virtual int BeforeAllTransformix( void );

  /** Execute stuff before the actual registration:
   * \li Set the initial transform and how to group transforms.
   */
  virtual void BeforeRegistrationBase( void );

  /** Execute stuff after the registration:
   * \li Get and set the final parameters for the resampler.
   */
  virtual void AfterRegistrationBase( void );

  /** Get the initial transform. */
  virtual const InitialTransformType * GetInitialTransform( void ) const;

  /** Set the initial transform. */
  virtual void SetInitialTransform( InitialTransformType * _arg );

  /** Set the TransformParametersFileName. */
  virtual void SetTransformParametersFileName( const char * filename );

  /** Get the TransformParametersFileName. */
  itkGetStringMacro( TransformParametersFileName );

  /** Function to read transform-parameters from a file. */
  virtual void ReadFromFile( void );

  /** Function to create transform-parameters map. */
  virtual void CreateTransformParametersMap(
    const ParametersType & param, ParameterMapType * paramsMap ) const;

  /** Function to write transform-parameters to a file. */
  virtual void WriteToFile( const ParametersType & param ) const;

  /** Function to write transform-parameters to a file. */
  virtual void WriteToFile( void ) const;

  /** Macro for reading and writing the transform parameters in WriteToFile or not. */
  virtual void SetReadWriteTransformParameters( const bool _arg );

  /** Function to read the initial transform parameters from a file. */
  virtual void ReadInitialTransformFromFile(
    const char * transformParameterFileName );

  /** Function to read the initial transform parameters from the internally stored
   * configuration object.
   */
  virtual void ReadInitialTransformFromVector( const size_t index );

  /** Function to transform coordinates from fixed to moving image. */
  virtual void TransformPoints( void ) const;

  /** Function to transform coordinates from fixed to moving image. */
  virtual void TransformPointsSomePoints( const std::string filename ) const;

  /** Function to transform coordinates from fixed to moving image, given as VTK file. */
  virtual void TransformPointsSomePointsVTK( const std::string filename ) const;

  /** Function to transform all coordinates from fixed to moving image. */
  virtual void TransformPointsAllPoints( void ) const;

  /** Function to compute the determinant of the spatial Jacobian. */
  virtual void ComputeDeterminantOfSpatialJacobian( void ) const;

  /** Function to compute the determinant of the spatial Jacobian. */
  virtual void ComputeSpatialJacobian( void ) const;

  /** Makes sure that the final parameters from the registration components
   * are copied, set, and stored.
   */
  virtual void SetFinalParameters( void );

protected:

  /** The constructor. */
  TransformBase();
  /** The destructor. */
  virtual ~TransformBase();

  /** Estimate a scales vector
   * AutomaticScalesEstimation works like this:
   * \li N=10000 points are sampled on a uniform grid on the fixed image.
   * \li Jacobians dT/dmu are computed
   * \li Scales_i = 1/N sum_x || dT / dmu_i ||^2
   */
  void AutomaticScalesEstimation( ScalesType & scales ) const;

  /** Member variables. */
  ParametersType * m_TransformParametersPointer;
  std::string      m_TransformParametersFileName;
  ParametersType   m_FinalParameters;

private:

  /** The private constructor. */
  TransformBase( const Self & );   // purposely not implemented
  /** The private copy constructor. */
  void operator=( const Self & );  // purposely not implemented

  /** Boolean to decide whether or not the transform parameters are written. */
  bool m_ReadWriteTransformParameters;

};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxTransformBase.hxx"
#endif

#endif // end #ifndef __elxTransformBase_h
