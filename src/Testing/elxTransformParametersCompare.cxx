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
/** \file
 \brief Compare two transform parameter files.

 Currently we only compare the transform parameter vector and not the fixed parameters.
 */
#include "itkCommandLineArgumentParser.h"
#include "itkParameterFileParser.h"
#include "itkParameterMapInterface.h"

#include "itkNumericTraits.h"
#include "itkMath.h"
#include "itksys/SystemTools.hxx"
#include "itkImage.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

/**
 * ******************* GetHelpString *******************
 */

std::string
GetHelpString( void )
{
  std::stringstream ss;
  ss << "Usage:" << std::endl
     << "elxTransformParametersCompare" << std::endl
     << "  -test      transform parameters file to test against baseline\n"
     << "  -base      baseline transform parameters filename\n"
     << "  [-mask]    mask image, only supported for the B-spline\n"
    //<< "  [-t]       intensity difference threshold, default 0\n"
     << "  [-a]       allowable tolerance (), default 1e-6\n"
     << "Computes (test - base) / base.";
  return ss.str();

} // end GetHelpString()


// This comparison works on all image types by reading images
// in a 4D double images. If images > 4 dimensions
// must be compared, change this variable.
static const unsigned int ITK_TEST_DIMENSION_MAX = 4;

int
main( int argc, char ** argv )
{
  /** Typedefs. */
  typedef itk::Image< float, ITK_TEST_DIMENSION_MAX >               CoefficientImageType;
  typedef CoefficientImageType::RegionType                          RegionType;
  typedef RegionType::SizeType                                      SizeType;
  typedef RegionType::IndexType                                     IndexType;
  typedef CoefficientImageType::SpacingType                         SpacingType;
  typedef CoefficientImageType::PointType                           PointType;
  typedef CoefficientImageType::PointType                           OriginType;
  typedef CoefficientImageType::DirectionType                       DirectionType;
  typedef itk::ImageRegionIteratorWithIndex< CoefficientImageType > IteratorType;
  typedef itk::ImageFileWriter< CoefficientImageType >              WriterType;

  typedef itk::Image< unsigned char, ITK_TEST_DIMENSION_MAX > MaskImageType;
  typedef itk::ImageFileReader< MaskImageType >               MaskReaderType;
  typedef itk::ImageRegionIteratorWithIndex< MaskImageType >  MaskIteratorType;

  /** Create command line argument parser. */
  itk::CommandLineArgumentParser::Pointer parser = itk::CommandLineArgumentParser::New();
  parser->SetCommandLineArguments( argc, argv );
  parser->SetProgramHelpText( GetHelpString() );

  parser->MarkArgumentAsRequired( "-test", "The input filename." );
  parser->MarkArgumentAsRequired( "-base", "The baseline filename." );

  itk::CommandLineArgumentParser::ReturnValue validateArguments = parser->CheckForRequiredArguments();

  if( validateArguments == itk::CommandLineArgumentParser::FAILED )
  {
    return EXIT_FAILURE;
  }
  else if( validateArguments == itk::CommandLineArgumentParser::HELPREQUESTED )
  {
    return EXIT_SUCCESS;
  }

  std::string testFileName;
  parser->GetCommandLineArgument( "-test", testFileName );

  std::string baselineFileName;
  parser->GetCommandLineArgument( "-base", baselineFileName );

  std::string maskFileName;
  parser->GetCommandLineArgument( "-mask", maskFileName );

  //double diffThreshold = 0.0;
  //parser->GetCommandLineArgument( "-t", diffThreshold );

  double allowedTolerance = 1e-6;
  parser->GetCommandLineArgument( "-a", allowedTolerance );

  if( allowedTolerance < 0 )
  {
    std::cerr << "ERROR: the specified allowed tolerance (-a) should be non-negative!" << std::endl;
    return EXIT_FAILURE;
  }

  /** Create parameter file reader. */
  typedef itk::ParameterFileParser   ParserType;
  typedef itk::ParameterMapInterface InterfaceType;

  typedef double ScalarType;
  std::string dummyErrorMessage = "";

  ParserType::Pointer    parameterFileParser = ParserType::New();
  InterfaceType::Pointer config              = InterfaceType::New();

  /** Read test parameters. */
  parameterFileParser->SetParameterFileName( testFileName.c_str() );
  try
  {
    parameterFileParser->ReadParameterFile();
  }
  catch( itk::ExceptionObject & err )
  {
    std::cerr << "Error during reading test transform parameters: " << err << std::endl;
    return EXIT_FAILURE;
  }

  config->SetParameterMap( parameterFileParser->GetParameterMap() );

  unsigned int numberOfParametersTest = 0;
  config->ReadParameter( numberOfParametersTest,
    "NumberOfParameters", 0, dummyErrorMessage );
  std::vector< ScalarType > parametersTest( numberOfParametersTest,
    itk::NumericTraits< ScalarType >::ZeroValue() );
  config->ReadParameter( parametersTest, "TransformParameters",
    0, numberOfParametersTest - 1, true, dummyErrorMessage );

  /** Read baseline parameters. */
  parameterFileParser->SetParameterFileName( baselineFileName.c_str() );
  try
  {
    parameterFileParser->ReadParameterFile();
  }
  catch( itk::ExceptionObject & err )
  {
    std::cerr << "Error during reading baseline transform parameters: " << err << std::endl;
    return EXIT_FAILURE;
  }
  config->SetParameterMap( parameterFileParser->GetParameterMap() );

  unsigned int numberOfParametersBaseline = 0;
  config->ReadParameter( numberOfParametersBaseline,
    "NumberOfParameters", 0, dummyErrorMessage );

  std::vector< ScalarType > parametersBaseline( numberOfParametersBaseline,
    itk::NumericTraits< ScalarType >::ZeroValue() );
  config->ReadParameter( parametersBaseline, "TransformParameters",
    0, numberOfParametersBaseline - 1, true, dummyErrorMessage );

  /** The sizes of the baseline and test parameters must match. */
  std::cerr << "Baseline transform parameters: " << baselineFileName
            << " has " << numberOfParametersBaseline << " parameters." << std::endl;
  std::cerr << "Test transform parameters:     " << testFileName
            << " has " << numberOfParametersTest << " parameters." << std::endl;

  if( numberOfParametersBaseline != numberOfParametersTest )
  {
    std::cerr << "ERROR: The number of transform parameters of the baseline and test do not match!" << std::endl;
    return EXIT_FAILURE;
  }

  /** Initialize variables. */
  ScalarType diffNorm           = itk::NumericTraits< ScalarType >::Zero;
  ScalarType baselineNorm       = itk::NumericTraits< ScalarType >::Zero;
  ScalarType diffNormNormalized = itk::NumericTraits< ScalarType >::Zero;

  /** Check if this is a B-spline transform.
   * If it is we write a sort of coefficient difference image.
   * Only the B-spline transform has mask support.
   */
  std::string transformName = "";
  config->ReadParameter( transformName, "Transform", 0, true, dummyErrorMessage );
  if( transformName != "BSplineTransform" )
  {
    /** Now compare the two parameter vectors. */
    for( unsigned int i = 0; i < numberOfParametersTest; i++ )
    {
      baselineNorm += vnl_math_sqr( parametersBaseline[ i ] );
      diffNorm     += vnl_math_sqr( parametersBaseline[ i ] - parametersTest[ i ] );
    }
    diffNormNormalized = vcl_sqrt( diffNorm ) / vcl_sqrt( baselineNorm );
  }
  else
  {
    /** This is a B-spline transform. Re-organize the parameters
     * as a coefficient image.
     */

    /** Get the true dimension. */
    unsigned int dimension = 2;
    config->ReadParameter( dimension, "FixedImageDimension", 0, true, dummyErrorMessage );

    /** Get coefficient image information. */
    SizeType      gridSize; gridSize.Fill( 1 );
    IndexType     gridIndex; gridIndex.Fill( 0 );
    SpacingType   gridSpacing; gridSpacing.Fill( 1.0 );
    OriginType    gridOrigin; gridOrigin.Fill( 0.0 );
    DirectionType gridDirection; gridDirection.SetIdentity();
    for( unsigned int i = 0; i < dimension; i++ )
    {
      config->ReadParameter( gridSize[ i ], "GridSize", i, true, dummyErrorMessage );
      config->ReadParameter( gridIndex[ i ], "GridIndex", i, true, dummyErrorMessage );
      config->ReadParameter( gridSpacing[ i ], "GridSpacing", i, true, dummyErrorMessage );
      config->ReadParameter( gridOrigin[ i ], "GridOrigin", i, true, dummyErrorMessage );
      for( unsigned int j = 0; j < dimension; j++ )
      {
        config->ReadParameter( gridDirection( j, i ),
          "GridDirection", i * dimension + j, true, dummyErrorMessage );
      }
    }

    /** Create the coefficient image. */
    CoefficientImageType::Pointer coefImage = CoefficientImageType::New();
    RegionType                    region; region.SetSize( gridSize ); region.SetIndex( gridIndex );
    coefImage->SetRegions( region );
    coefImage->SetSpacing( gridSpacing );
    coefImage->SetOrigin( gridOrigin );
    coefImage->SetDirection( gridDirection );
    coefImage->Allocate();

    /** Read the mask image, if given. */
    MaskImageType::Pointer maskImage;
    MaskIteratorType       itM;
    if( maskFileName != "" )
    {
      MaskReaderType::Pointer maskReader = MaskReaderType::New();
      maskReader->SetFileName( maskFileName );
      maskReader->Update();
      maskImage = maskReader->GetOutput();

      itM = MaskIteratorType( maskImage, maskImage->GetLargestPossibleRegion() );
    }

    /** Fill the coefficient image with the difference of the B-spline
     * parameters. Since there are dimension number of differences,
     * we take the norm.
     */
    IteratorType       it( coefImage, coefImage->GetLargestPossibleRegion() );
    unsigned int       index = 0; float include = 0.0; bool isInside = false;
    IndexType          indexInCoefficientImage;
    IndexType          indexInMaskImage;
    PointType          physicalPoint;
    const unsigned int numberParPerDim = numberOfParametersTest / dimension;
    while( !it.IsAtEnd() )
    {
      /** Voxel content. */
      ScalarType diffNormTmp = itk::NumericTraits< ScalarType >::Zero;
      for( unsigned int i = 0; i < dimension; i++ )
      {
        unsigned int j = index + i * numberParPerDim;
        diffNormTmp += vnl_math_sqr( parametersBaseline[ j ] - parametersTest[ j ] );
      }
      diffNormTmp = vcl_sqrt( diffNormTmp );
      it.Set( diffNormTmp );

      /** Compare. */
      include = 1.0;
      if( maskFileName != "" )
      {
        indexInCoefficientImage = it.GetIndex();
        coefImage->TransformIndexToPhysicalPoint( indexInCoefficientImage, physicalPoint );
        isInside = maskImage->TransformPhysicalPointToIndex( physicalPoint, indexInMaskImage );
        itM.SetIndex( indexInMaskImage );
        if( isInside && itM.Value() == 0 ) { include = 0.0; }
      } // end mask

      for( unsigned int i = 0; i < dimension; i++ )
      {
        unsigned int j = index + i * numberParPerDim;
        baselineNorm += include * vnl_math_sqr( parametersBaseline[ j ] );
        diffNorm     += include * vnl_math_sqr( parametersBaseline[ j ] - parametersTest[ j ] );
      }

      /** Update iterators. */
      ++it; ++index;
      if( maskFileName != "" ) { ++itM; }
    } // end while

    /** Final normalized norm. */
    diffNormNormalized = vcl_sqrt( diffNorm ) / vcl_sqrt( baselineNorm );

    /** Create name for difference image. */
    std::string diffImageFileName
      = itksys::SystemTools::GetFilenamePath( testFileName );
    if( diffImageFileName != "" ) { diffImageFileName += "/"; }
    diffImageFileName
                      += itksys::SystemTools::GetFilenameWithoutLastExtension( testFileName );
    diffImageFileName += "_DIFF_PAR.mha";

    /** Write the difference image. */
    if( diffNormNormalized > 1e-10 )
    {
      WriterType::Pointer writer = WriterType::New();
      writer->SetFileName( diffImageFileName );
      writer->SetInput( coefImage );
      try
      {
        writer->Write();
      }
      catch( itk::ExceptionObject & err )
      {
        std::cerr << "Error during writing difference image: " << err << std::endl;
        return EXIT_FAILURE;
      }
    }
  } // end B-spline

  /** Print result to screen. */
  std::cerr << "The norm of the difference between baseline and test "
            << "transform parameters was computed, using\n"
            << "    || baseline - test ||\n"
            << "    ---------------------\n"
            << "       || baseline ||\n";
  std::cerr << "Computed difference: "
            << vcl_sqrt( diffNorm ) << " / " << vcl_sqrt( baselineNorm ) << " = "
            << diffNormNormalized << std::endl;
  std::cerr << "Allowed  difference: " << allowedTolerance << std::endl;

  /** Return. */
  if( diffNormNormalized > allowedTolerance )
  {
    std::cerr << "ERROR: The difference is larger than acceptable!" << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;

} // end main
