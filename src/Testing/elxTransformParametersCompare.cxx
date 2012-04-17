/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
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
#include "itkImageRegionIterator.h"
#include "itkImageFileWriter.h"


/**
 * ******************* GetHelpString *******************
 */

std::string GetHelpString( void )
{
  std::stringstream ss;
  ss << "Usage:" << std::endl
    << "elxTransformParametersCompare" << std::endl
    << "  -test      transform parameters file to test against baseline\n"
    << "  -base      baseline transform parameters filename\n"
    //<< "  [-t]       intensity difference threshold, default 0\n"
    << "  [-a]       allowable tolerance (), default 1e-6";
  return ss.str();

} // end GetHelpString()

// This comparison works on all image types by reading images in a 6D double images. If images > 6 dimensions
// must be compared, change this variable.
static const unsigned int ITK_TEST_DIMENSION_MAX = 4;

int main( int argc, char **argv )
{
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

  //double diffThreshold = 0.0;
  //parser->GetCommandLineArgument( "-t", diffThreshold );

  double allowedTolerance = 1e-6;
  parser->GetCommandLineArgument( "-a", allowedTolerance );

  if ( allowedTolerance < 0 )
  {
    std::cerr << "ERROR: the specified allowed tolerance (-a) should be non-negative!" << std::endl;
    return EXIT_FAILURE;
  }

  /** Create parameter file reader. */
  typedef itk::ParameterFileParser    ParserType;
  typedef itk::ParameterMapInterface  InterfaceType;

  typedef double ScalarType;
  std::string dummyErrorMessage = "";

  ParserType::Pointer parameterFileParser = ParserType::New();
  InterfaceType::Pointer config = InterfaceType::New();

  /** Read test parameters. */
  parameterFileParser->SetParameterFileName( testFileName.c_str() );
  try
  {
    parameterFileParser->ReadParameterFile();
  }
  catch ( itk::ExceptionObject & err )
  {
    std::cerr << "Error during reading test transform parameters: " << err << std::endl;
    return EXIT_FAILURE;
  }

  config->SetParameterMap( parameterFileParser->GetParameterMap() );

  unsigned int numberOfParametersTest = 0;
  config->ReadParameter( numberOfParametersTest,
    "NumberOfParameters", 0, dummyErrorMessage );
  std::vector<ScalarType> parametersTest( numberOfParametersTest,
    itk::NumericTraits<ScalarType>::Zero );
  config->ReadParameter( parametersTest, "TransformParameters",
    0, numberOfParametersTest - 1, true, dummyErrorMessage );

  /** Read baseline parameters. */
  parameterFileParser->SetParameterFileName( baselineFileName.c_str() );
  try
  {
    parameterFileParser->ReadParameterFile();
  }
  catch ( itk::ExceptionObject & err )
  {
    std::cerr << "Error during reading baseline transform parameters: " << err << std::endl;
    return EXIT_FAILURE;
  }
  config->SetParameterMap( parameterFileParser->GetParameterMap() );

  unsigned int numberOfParametersBaseline = 0;
  config->ReadParameter( numberOfParametersBaseline,
    "NumberOfParameters", 0, dummyErrorMessage );

  std::vector<ScalarType> parametersBaseline( numberOfParametersBaseline,
    itk::NumericTraits<ScalarType>::Zero );
  config->ReadParameter( parametersBaseline, "TransformParameters",
    0, numberOfParametersBaseline - 1, true, dummyErrorMessage );

  /** The sizes of the baseline and test parameters must match. */
  std::cerr << "Baseline transform parameters: " << baselineFileName
    << " has " << numberOfParametersBaseline << " parameters." << std::endl;
  std::cerr << "Test transform parameters:     " << testFileName
    << " has " << numberOfParametersTest << " parameters." << std::endl;

  if ( numberOfParametersBaseline != numberOfParametersTest )
  {
    std::cerr << "ERROR: The number of transform parameters of the baseline and test do not match!" << std::endl;
    return EXIT_FAILURE;
  }

  /** Now compare the two parameter vectors. */
  ScalarType diffNorm = itk::NumericTraits<ScalarType>::Zero;
  for ( unsigned int i = 0; i < numberOfParametersTest; i++ )
  {
    diffNorm += vnl_math_sqr( parametersBaseline[ i ] - parametersTest[ i ] );
  }
  diffNorm = vcl_sqrt( diffNorm );

  std::cerr << "The norm of the difference between baseline and test transform parameters was computed\n";
  std::cerr << "Computed difference: " << diffNorm << std::endl;
  std::cerr << "Allowed  difference: " << allowedTolerance << std::endl;

  /** Check if this is a B-spline transform.
   * If it is and if the difference is nonzero, we write a sort of coefficient
   * difference image.
   */
  std::string transformName = "";
  config->ReadParameter( transformName, "Transform", 0, true, dummyErrorMessage );
  if( diffNorm > 1e-18 && transformName == "BSplineTransform" )
  {
    /** Get the true dimension. */
    unsigned int dimension = 2;
    config->ReadParameter( dimension, "FixedImageDimension", 0, true, dummyErrorMessage );

    /** Typedefs. */
    typedef itk::Image< float, ITK_TEST_DIMENSION_MAX >     CoefficientImageType;
    typedef CoefficientImageType::RegionType                RegionType;
    typedef RegionType::SizeType                            SizeType;
    typedef RegionType::IndexType                           IndexType;
    typedef CoefficientImageType::SpacingType               SpacingType;
    typedef CoefficientImageType::PointType                 OriginType;
    typedef CoefficientImageType::DirectionType             DirectionType;
    typedef itk::ImageRegionIterator<CoefficientImageType>  IteratorType;
    typedef itk::ImageFileWriter<CoefficientImageType>      WriterType;

    /** Get coefficient image information. */
    SizeType gridSize; gridSize.Fill( 1 );
    IndexType gridIndex; gridIndex.Fill( 0 );
    SpacingType gridSpacing; gridSpacing.Fill( 1.0 );
    OriginType gridOrigin; gridOrigin.Fill( 0.0 );
    DirectionType gridDirection; gridDirection.SetIdentity();
    for ( unsigned int i = 0; i < dimension; i++ )
    {
      config->ReadParameter( gridSize[ i ], "GridSize", i, true, dummyErrorMessage );
      config->ReadParameter( gridIndex[ i ], "GridIndex", i, true, dummyErrorMessage );
      config->ReadParameter( gridSpacing[ i ], "GridSpacing", i, true, dummyErrorMessage );
      config->ReadParameter( gridOrigin[ i ], "GridOrigin", i, true, dummyErrorMessage );
      for ( unsigned int j = 0; j < dimension; j++ )
      {
        config->ReadParameter( gridDirection( j, i),
          "GridDirection", i * dimension + j, true, dummyErrorMessage );
      }
    }

    /** Create the coefficient image. */
    CoefficientImageType::Pointer coefImage = CoefficientImageType::New();
    RegionType region; region.SetSize( gridSize ); region.SetIndex( gridIndex );
    coefImage->SetRegions( region );
    coefImage->SetSpacing( gridSpacing );
    coefImage->SetOrigin( gridOrigin );
    coefImage->SetDirection( gridDirection );
    coefImage->Allocate();

    /** Fill the coefficient image with the difference of the B-spline
     * parameters. Since there are dimension number of differences,
     * we take the norm.
     */
    IteratorType it( coefImage, coefImage->GetLargestPossibleRegion() );
    it.GoToBegin();
    unsigned int index = 0;
    const unsigned int numberParPerDim = numberOfParametersTest / dimension;
    while( !it.IsAtEnd() )
    {
      ScalarType diffNorm = itk::NumericTraits<ScalarType>::Zero;
      for ( unsigned int i = 0; i < dimension; i++ )
      {
        unsigned int j = index + i * numberParPerDim;
        diffNorm += vnl_math_sqr( parametersBaseline[ j ] - parametersTest[ j ] );
      }
      diffNorm = vcl_sqrt( diffNorm );

      it.Set( diffNorm );
      ++it; index++;
    }

    /** Create name for difference image. */
    std::string diffImageFileName =
      itksys::SystemTools::GetFilenamePath( testFileName );
    diffImageFileName += "/";
    diffImageFileName +=
      itksys::SystemTools::GetFilenameWithoutLastExtension( testFileName );
    diffImageFileName += "_DIFF_PAR.mha";

    /** Write the difference image. */
    WriterType::Pointer writer = WriterType::New();
    writer->SetFileName( diffImageFileName );
    writer->SetInput( coefImage );
    try
    {
      writer->Write();
    }
    catch ( itk::ExceptionObject & err )
    {
      std::cerr << "Error during writing difference image: " << err << std::endl;
      return EXIT_FAILURE;
    }
  }

  /** Return. */
  if( diffNorm > allowedTolerance )
  {
    std::cerr << "ERROR: The difference is larger than acceptable!" << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;

} // end main

