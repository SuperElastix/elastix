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
#include "itkCommandLineArgumentParser.h"
#include "itkParameterFileParser.h"
#include "itkParameterMapInterface.h"

#include "itkImage.h"
#include "itkImageFileReader.h"

// Supported transforms:
#include "itkTransform.h"
#include "itkEulerTransform.h"
//#include "itkEuler3DTransform.h"
#include "itkAffineTransform.h"

#include <iostream>
#include <iomanip>

/**
 * ******************* GetHelpString *******************
 */

std::string
GetHelpString( void )
{
  std::stringstream ss;
  ss << "Usage:" << std::endl
     << "elxInvertTransform" << std::endl
     << "  -tp    transform parameters file to be inverted\n"
     << "  -out   output inverted transform parameters filename\n"
     << "  -m     moving image file name\n"
     << "Currently only 3D, {Euler, Affine} supported.";
  return ss.str();

} // end GetHelpString()


int
main( int argc, char * argv[] )
{
  /** Read the command line arguments. */
  itk::CommandLineArgumentParser::Pointer clParser = itk::CommandLineArgumentParser::New();
  clParser->SetCommandLineArguments( argc, argv );
  clParser->SetProgramHelpText( GetHelpString() );

  clParser->MarkArgumentAsRequired( "-tp", "The input transform parameter file." );
  clParser->MarkArgumentAsRequired( "-out", "The output transform parameter file." );
  clParser->MarkArgumentAsRequired( "-m", "The moving image." );

  itk::CommandLineArgumentParser::ReturnValue validateArguments = clParser->CheckForRequiredArguments();

  if( validateArguments == itk::CommandLineArgumentParser::FAILED )
  {
    return EXIT_FAILURE;
  }
  else if( validateArguments == itk::CommandLineArgumentParser::HELPREQUESTED )
  {
    return EXIT_SUCCESS;
  }

  std::string inputTransformParametersName = "";
  clParser->GetCommandLineArgument( "-tp", inputTransformParametersName );

  std::string outputTransformParametersName = "";
  clParser->GetCommandLineArgument( "-out", outputTransformParametersName );

  std::string movingImageFileName;
  clParser->GetCommandLineArgument( "-m", movingImageFileName );

  /** Typedef's. */
  //const unsigned int Dimension = 2;
  const unsigned int Dimension = 3;
  typedef float PrecisionType;
  std::string dummyErrorMessage = "";

  typedef itk::Transform<
    PrecisionType, Dimension, Dimension >                  BaseTransformType;
  typedef itk::EulerTransform< PrecisionType, Dimension >  RigidTransformType;
  typedef itk::AffineTransform< PrecisionType, Dimension > AffineTransformType;
  typedef BaseTransformType::ParametersType                ParametersType;
  typedef BaseTransformType::ScalarType                    ScalarType;
  typedef RigidTransformType::CenterType                   CenterType;
  //typedef BaseTransformType::OutputPointType               OutputPointType;

  /** Interface to the original transform parameters file. */
  typedef itk::ParameterFileParser   ParserType;
  typedef itk::ParameterMapInterface InterfaceType;
  ParserType::Pointer    parser = ParserType::New();
  InterfaceType::Pointer config = InterfaceType::New();
  parser->SetParameterFileName( inputTransformParametersName );
  parser->ReadParameterFile();
  config->SetParameterMap( parser->GetParameterMap() );

  /** Check no initial transform. */
  std::string initialTransform = "";
  config->ReadParameter( initialTransform, "InitialTransformParametersFileName", 0, dummyErrorMessage );
  if( initialTransform != "NoInitialTransform" )
  {
    std::cerr << "ERROR: currently only a single non-concatenated transform is supported!\n"
              << "  The parameter \"InitialTransformParametersFileName\" should read \"NoInitialTransform\"." << std::endl;
    return EXIT_FAILURE;
  }

  /** Check dimension. */
  unsigned int dimF = 0;
  config->ReadParameter( dimF, "FixedImageDimension", 0, dummyErrorMessage );
  if( dimF != Dimension )
  {
    std::cerr << "ERROR: the program elxInvertTransform was compiled for images of dimension " << Dimension << ",\n"
              << "  while the parameter \"FixedImageDimension\" reads " << dimF << ".\n"
              << "  Recompile elxInvertTransform for Dimension = " << dimF << std::endl;
    return EXIT_FAILURE;
  }

  /**
   * *** TASK 1:
   * *** Read the moving image to determine the inverted fixed image domain.
   * *** Needed to determine the parameters Size, Index, Spacing, Direction in
   * *** the output inverted transform parameter file.
   */

  /** Create a testReader. */
  typedef itk::Image< short, Dimension >         DummyImageType;
  typedef itk::ImageFileReader< DummyImageType > ReaderType;
  ReaderType::Pointer testReader = ReaderType::New();
  testReader->SetFileName( movingImageFileName.c_str() );

  /** Generate all information. */
  try
  {
    testReader->UpdateOutputInformation();
  }
  catch( itk::ExceptionObject & e )
  {
    std::cerr << "ERROR: Caught ITK exception: " << e << std::endl;
    return EXIT_FAILURE;
  }
  itk::ImageIOBase::Pointer imageIOBase = testReader->GetImageIO();

  /**
   * *** TASK 2:
   * *** Read the original transform parameters to setup the original transform
   * *** and invert the transform.
   */

  /** Get the number of TransformParameters. */
  unsigned int numberOfParameters = 0;
  config->ReadParameter( numberOfParameters, "NumberOfParameters", 0, dummyErrorMessage );

  /** Read the TransformParameters as a vector. */
  std::vector< ScalarType > vecPar( numberOfParameters,
    itk::NumericTraits< ScalarType >::ZeroValue() );
  config->ReadParameter( vecPar, "TransformParameters",
    0, numberOfParameters - 1, true, dummyErrorMessage );

  /** Convert to ParametersType. */
  ParametersType transformParameters( numberOfParameters );
  for( unsigned int i = 0; i < numberOfParameters; i++ )
  {
    transformParameters[ i ] = vecPar[ i ];
  }

  /** Get center of rotation. */
  CenterType centerOfRotation;
  for( unsigned int i = 0; i < Dimension; i++ )
  {
    config->ReadParameter( centerOfRotation[ i ],
      "CenterOfRotationPoint", i, dummyErrorMessage );
  }

  /** Set up the transform. */
  ParametersType transformParametersInv( numberOfParameters );
  CenterType     centerOfRotationInv;

  std::string transformType = "";
  config->ReadParameter( transformType, "Transform", 0, dummyErrorMessage );

  try
  {
    if( transformType == "EulerTransform" )
    {
      RigidTransformType::Pointer rigidTransform = RigidTransformType::New();
      rigidTransform->SetCenter( centerOfRotation );
      rigidTransform->SetParametersByValue( transformParameters );

      RigidTransformType::Pointer inverseRigidTransform = RigidTransformType::New();
      rigidTransform->GetInverse( inverseRigidTransform );

      transformParametersInv = inverseRigidTransform->GetParameters();
      centerOfRotationInv    = inverseRigidTransform->GetCenter();
    }
    else if( transformType == "AffineTransform" )
    {
      AffineTransformType::Pointer affineTransform = AffineTransformType::New();
      affineTransform->SetCenter( centerOfRotation );
      affineTransform->SetParametersByValue( transformParameters );

      AffineTransformType::Pointer inverseAffineTransform = AffineTransformType::New();
      affineTransform->GetInverse( inverseAffineTransform );

      transformParametersInv = inverseAffineTransform->GetParameters();
      centerOfRotationInv    = inverseAffineTransform->GetCenter();
    }
    else
    {
      std::cerr << "ERROR: Transforms of the type "
                << transformType
                << " are not supported." << std::endl;
      return EXIT_FAILURE;
    }
  }
  catch( itk::ExceptionObject & e )
  {
    std::cerr << "ERROR: Caught ITK exception: " << e << std::endl;
    return EXIT_FAILURE;
  }

  /**
   * *** TASK 3:
   * *** Write the inverted transform to file, in elastix style.
   */

  /** Get some information from the input transform parameters file that needs to be copied. */
  std::string combinationMethod = "Compose";
  config->ReadParameter( combinationMethod, "HowToCombineTransforms", 0, dummyErrorMessage );

  unsigned int FixDim = Dimension;
  config->ReadParameter( FixDim, "FixedImageDimension", 0, dummyErrorMessage );
  unsigned int MovDim = Dimension;
  config->ReadParameter( MovDim, "MovingImageDimension", 0, dummyErrorMessage );

  std::string fixpix = "float";
  config->ReadParameter( fixpix, "FixedInternalImagePixelType", 0, dummyErrorMessage );
  std::string movpix = "float";
  config->ReadParameter( movpix, "MovingInternalImagePixelType", 0, dummyErrorMessage );

  std::string useDirectionCosines = "true";
  config->ReadParameter( useDirectionCosines, "UseDirectionCosines", 0, dummyErrorMessage );

  /** Open a file for writing. */
  std::ofstream outputTPFile( outputTransformParametersName.c_str() );

  /** The following is a modified copy of elx::TransformBase::WriteToFile(). */

  /** Write the name of this transform. */
  outputTPFile << "(Transform \""
               << transformType << "\")" << std::endl;

  /** Write the number of parameters of this transform. */
  outputTPFile << "(NumberOfParameters "
               << numberOfParameters << ")" << std::endl;

  /** In this case, write in a normal way to the parameter file. */
  outputTPFile << "(TransformParameters ";
  for( unsigned int i = 0; i < numberOfParameters - 1; i++ )
  {
    outputTPFile << transformParametersInv[ i ] << " ";
  }
  outputTPFile << transformParametersInv[ numberOfParameters - 1 ] << ")" << std::endl;

  /** Write the name of the parameters-file of the initial transform. */
  outputTPFile << "(InitialTransformParametersFileName \"NoInitialTransform\")" << std::endl;

  /** Write the way Transforms are combined. */
  outputTPFile << "(HowToCombineTransforms \""
               << combinationMethod << "\")" << std::endl;

  /** Write image specific things. */
  outputTPFile << std::endl << "// Image specific" << std::endl;

  /** Write image dimensions. */
  outputTPFile << "(FixedImageDimension "
               << FixDim << ")" << std::endl;
  outputTPFile << "(MovingImageDimension "
               << MovDim << ")" << std::endl;

  /** Write image pixel types. */
  outputTPFile << "(FixedInternalImagePixelType \""
               << fixpix << "\")" << std::endl;
  outputTPFile << "(MovingInternalImagePixelType \""
               << movpix << "\")" << std::endl;

  /** Get the Size, Spacing and Origin of the moving image. */

  /** Write image Size. */
  outputTPFile << "(Size ";
  for( unsigned int i = 0; i < MovDim - 1; i++ )
  {
    outputTPFile << imageIOBase->GetDimensions( i ) << " ";
  }
  outputTPFile << imageIOBase->GetDimensions( MovDim - 1 ) << ")" << std::endl;

  /** Write image Index. */
  outputTPFile << "(Index";
  for( unsigned int i = 0; i < MovDim; i++ )
  {
    outputTPFile << " 0";
  }
  outputTPFile << ")" << std::endl;

  /** Set the precision of cout to 10, because Spacing and
   * Origin must have at least one digit precision.
   */
  outputTPFile << std::setprecision( 10 );

  /** Write image Spacing. */
  outputTPFile << "(Spacing ";
  for( unsigned int i = 0; i < MovDim - 1; i++ )
  {
    outputTPFile << imageIOBase->GetSpacing( i ) << " ";
  }
  outputTPFile << imageIOBase->GetSpacing( MovDim - 1 ) << ")" << std::endl;

  /** Write image Origin. */
  outputTPFile << "(Origin ";
  for( unsigned int i = 0; i < MovDim - 1; i++ )
  {
    outputTPFile << imageIOBase->GetOrigin( i ) << " ";
  }
  outputTPFile << imageIOBase->GetOrigin( MovDim - 1 ) << ")" << std::endl;

  /** Write direction cosines. */
  outputTPFile << "(Direction";
  for( unsigned int i = 0; i < MovDim; i++ )
  {
    for( unsigned int j = 0; j < MovDim; j++ )
    {
      outputTPFile << " " << imageIOBase->GetDirection( i )[ j ];
    }
  }
  outputTPFile << ")" << std::endl;

  /** Set the precision back to default value. */
  outputTPFile << std::setprecision( 6 );

  /** Write whether the direction cosines should be taken into account.
   * This parameter is written from elastix 4.203.
   */
  outputTPFile << "(UseDirectionCosines \""
               << useDirectionCosines << "\")" << std::endl;

  /** END elx::TransformBase::WriteToFile(). */

  /** Read from input transform parameter file. */
  std::string computeZYX = "true";
  config->ReadParameter( computeZYX, "ComputeZYX", 0, dummyErrorMessage );

  std::string resampleInterpolator = "FinalBSplineInterpolator";
  config->ReadParameter( resampleInterpolator, "ResampleInterpolator", 0, dummyErrorMessage );

  unsigned int interpolationOrder = 3;
  config->ReadParameter( interpolationOrder, "FinalBSplineInterpolationOrder", 0, dummyErrorMessage );

  std::string resampler = "DefaultResampler";
  config->ReadParameter( resampler, "DefaultResampler", 0, dummyErrorMessage );

  float defaultPixelValue = 0.0;
  config->ReadParameter( defaultPixelValue, "DefaultPixelValue", 0, dummyErrorMessage );

  std::string resultImageFormat = "mhd";
  config->ReadParameter( resultImageFormat, "ResultImageFormat", 0, dummyErrorMessage );

  std::string resultImagePixelType = "short";
  config->ReadParameter( resultImagePixelType, "ResultImagePixelType", 0, dummyErrorMessage );

  std::string compressResultImage = "false";
  config->ReadParameter( compressResultImage, "CompressResultImage", 0, dummyErrorMessage );

  /** Write to file. */
  outputTPFile << "\n// " << transformType << " specific\n";
  outputTPFile << "(CenterOfRotationPoint";
  for( unsigned int i = 0; i < Dimension; i++ )
  {
    outputTPFile << " " << centerOfRotationInv[ i ];
  }
  outputTPFile << ")\n";
  outputTPFile << "(ComputeZYX \"" << computeZYX << "\")\n";

  outputTPFile << "\n// ResampleInterpolator specific\n";
  outputTPFile << "(ResampleInterpolator \"" << resampleInterpolator << "\")\n";
  outputTPFile << "(FinalBSplineInterpolationOrder " << interpolationOrder << ")\n"; // assuming B-spline here

  outputTPFile << "\n// Resampler specific\n";
  outputTPFile << "(Resampler \"" << resampler << "\")\n";
  outputTPFile << "(DefaultPixelValue " << defaultPixelValue << ")\n";
  outputTPFile << "(ResultImageFormat \"" << resultImageFormat << "\")\n";
  outputTPFile << "(ResultImagePixelType \"" << resultImagePixelType << "\")\n";
  outputTPFile << "(CompressResultImage \"" << compressResultImage << "\")\n";

  // It would be better to copy everything else from the input

  outputTPFile.close();

  return EXIT_SUCCESS;

} // end main()
