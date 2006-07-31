#ifndef __elxKNNGraphMultiDimensionalAlphaMutualInformationMetric_HXX__
#define __elxKNNGraphMultiDimensionalAlphaMutualInformationMetric_HXX__

#include "elxKNNGraphMultiDimensionalAlphaMutualInformationMetric.h"

#include "itkImageFileReader.h"
#include "itkBSplineInterpolateImageFunction.h"

#include <string>


namespace elastix
{
using namespace itk;

	/**
	 * ******************* Initialize ***********************
	 */

	template <class TElastix>
		void KNNGraphMultiDimensionalAlphaMutualInformationMetric<TElastix>
		::Initialize(void) throw (ExceptionObject)
	{
		TimerPointer timer = TimerType::New();
		timer->StartTimer();
		this->Superclass1::Initialize();
		timer->StopTimer();
		elxout << "Initialization of KNNGraphAlphaMutualInformation metric took: "
			<< static_cast<long>( timer->GetElapsedClockSec() * 1000 ) << " ms." << std::endl;

	} // end Initialize

	
  /**
	 * ***************** BeforeRegistration ***********************
	 */

	template <class TElastix>
		void KNNGraphMultiDimensionalAlphaMutualInformationMetric<TElastix>
		::BeforeRegistration(void)
	{
    /** Get and set alpha, from alpha - MI. */
		double alpha = 0.5;
    this->m_Configuration->ReadParameter( alpha, "Alpha", 0 );
		this->SetAlpha( alpha );

    /** Get and set the number of fixed feature images. */
    unsigned int noFFI = 0;
    this->m_Configuration->ReadParameter( noFFI, "NumberOfFixedFeatureImages", 0 );
    this->SetNumberOfFixedFeatureImages( noFFI );

    /** Get and set the number of moving feature images. */
    unsigned int noMFI = 0;
    this->m_Configuration->ReadParameter( noMFI, "NumberOfMovingFeatureImages", 0 );
    this->SetNumberOfMovingFeatureImages( noMFI );

    /** Get the filenames of the fixed feature images. */
    std::vector< std::string > filenamesFFI( noFFI );
    for ( unsigned int i = 0; i < noFFI; i++ )
    {
      /** Get the i-th filename. */
      unsigned int ret = this->m_Configuration->ReadParameter(
        filenamesFFI[ i ], "FixedFeatureImageFileNames", i );
      /** It must be given. */
      if ( ret )
      {
        itkExceptionMacro( << "ERROR: No filename for fixed feature image " << i << " given." );
      }
    }

    /** Get the filenames of the moving feature images. */
    std::vector< std::string > filenamesMFI( noMFI );
    for ( unsigned int i = 0; i < noMFI; i++ )
    {
      /** Get the i-th filename. */
      unsigned int ret = this->m_Configuration->ReadParameter(
        filenamesMFI[ i ], "MovingFeatureImageFileNames", i );
      /** It must be given. */
      if ( ret )
      {
        itkExceptionMacro( << "ERROR: No filename for moving feature image " << i << " given." );
      }
    }

    /** Read and set the fixed feature images. */
    typedef ImageFileReader< FixedFeatureImageType >                FixedFeatureReaderType;
    typedef std::vector< typename FixedFeatureReaderType::Pointer > FixedFeatureVectorReaderType;
    FixedFeatureVectorReaderType readersFFI( noFFI );
    for ( unsigned int i = 0; i < noFFI; i++ )
    {
      readersFFI[ i ] = FixedFeatureReaderType::New();
      readersFFI[ i ]->SetFileName( filenamesFFI[ i ].c_str() );
      readersFFI[ i ]->Update();
      this->SetFixedFeatureImage( i, readersFFI[ i ]->GetOutput() );
    }

    /** Read and set the moving feature images. */
    typedef ImageFileReader< MovingFeatureImageType >                 MovingFeatureReaderType;
    typedef std::vector< typename MovingFeatureReaderType::Pointer >  MovingFeatureVectorReaderType;
    MovingFeatureVectorReaderType readersMFI( noMFI );
    for ( unsigned int i = 0; i < noMFI; i++ )
    {
      readersMFI[ i ] = MovingFeatureReaderType::New();
      readersMFI[ i ]->SetFileName( filenamesMFI[ i ].c_str() );
      readersMFI[ i ]->Update();
      this->SetMovingFeatureImage( i, readersMFI[ i ]->GetOutput() );
    }

    /** Get the spline order of the fixed feature image interpolators. */
    unsigned int splineOrder = 1;
    this->m_Configuration->ReadParameter(
      splineOrder, "FixedFeatureInterpolatorBSplineOrder", 0 );
    std::vector< unsigned int > soFFII( noFFI, splineOrder);
    for ( unsigned int i = 1; i < noFFI; i++ )
    {
      this->m_Configuration->ReadParameter( 
        soFFII[ i ], "FixedFeatureInterpolatorBSplineOrder", i, true );
    }

    /** Get the spline order of the moving feature image interpolators. */
    splineOrder = 1;
    this->m_Configuration->ReadParameter(
      splineOrder, "MovingFeatureInterpolatorBSplineOrder", 0 );
    std::vector< unsigned int > soMFII( noMFI, splineOrder);
    for ( unsigned int i = 1; i < noMFI; i++ )
    {
      this->m_Configuration->ReadParameter(
        soMFII[ i ], "MovingFeatureInterpolatorBSplineOrder", i, true );
    }

    /** Create and set interpolators for the fixed feature images. */
    typedef BSplineInterpolateImageFunction< FixedFeatureImageType >        FixedFeatureInterpolatorType;
    typedef std::vector< typename FixedFeatureInterpolatorType::Pointer >   FixedFeatureInterpolatorVectorType;
    FixedFeatureInterpolatorVectorType interpolatorsFFI( noFFI );
    for ( unsigned int i = 0; i < noFFI; i++ )
    {
      interpolatorsFFI[ i ] = FixedFeatureInterpolatorType::New();
      interpolatorsFFI[ i ]->SetSplineOrder( soFFII[ i ] );
      this->SetFixedFeatureInterpolator( i, interpolatorsFFI[ i ] );
    }

    /** Create and set interpolators for the moving feature images. */
    typedef BSplineInterpolateImageFunction< MovingFeatureImageType >       MovingFeatureInterpolatorType;
    typedef std::vector< typename MovingFeatureInterpolatorType::Pointer >  MovingFeatureInterpolatorVectorType;
    MovingFeatureInterpolatorVectorType interpolatorsMFI( noMFI );
    for ( unsigned int i = 0; i < noMFI; i++ )
    {
      interpolatorsMFI[ i ] = MovingFeatureInterpolatorType::New();
      interpolatorsMFI[ i ]->SetSplineOrder( soMFII[ i ] );
      this->SetMovingFeatureInterpolator( i, interpolatorsMFI[ i ] );
    }

  } // end BeforeRegistration


	/**
	 * ***************** BeforeEachResolution ***********************
	 */

	template <class TElastix>
		void KNNGraphMultiDimensionalAlphaMutualInformationMetric<TElastix>
		::BeforeEachResolution(void)
	{
		/** Get the current resolution level. */
		unsigned int level = 
			( this->m_Registration->GetAsITKBaseType() )->GetCurrentLevel();

    /** Get the parameters for the KNN binary tree. */

    /** Get the tree type. */
    std::string treeType = "KDTree";
    this->m_Configuration->ReadParameter( treeType, "TreeType", 0 );
		this->m_Configuration->ReadParameter( treeType, "TreeType", level, true );

    bool silentBS = false;
    bool silentSplit = false;
    bool silentShrink = false;
    if ( treeType == "KDTree" )
    {
      silentShrink = true;
    }
    else if ( treeType == "BruteForceTree" )
    {
      silentBS = true;
      silentSplit = true;
      silentShrink = true;
    }

    /** Get the bucket size. */
		unsigned int bucketSize = 50;
    this->m_Configuration->ReadParameter( bucketSize, "BucketSize", 0, silentBS );
		this->m_Configuration->ReadParameter( bucketSize, "BucketSize", level, true );

    /** Get the splitting rule for all trees. */
    std::string splittingRule = "ANN_KD_SL_MIDPT";
    this->m_Configuration->ReadParameter( splittingRule, "SplittingRule", 0, silentSplit );
		this->m_Configuration->ReadParameter( splittingRule, "SplittingRule", level, true );

    /** Get the splitting rule for the fixed tree. */
    std::string fixedSplittingRule = "ANN_KD_SL_MIDPT";
    this->m_Configuration->ReadParameter( fixedSplittingRule, "FixedSplittingRule", 0, silentSplit );
		this->m_Configuration->ReadParameter( fixedSplittingRule, "FixedSplittingRule", level, true );

    /** Get the splitting rule for the moving tree. */
    std::string movingSplittingRule = "ANN_KD_SL_MIDPT";
    this->m_Configuration->ReadParameter( movingSplittingRule, "MovingSplittingRule", 0, silentSplit );
		this->m_Configuration->ReadParameter( movingSplittingRule, "MovingSplittingRule", level, true );

    /** Get the splitting rule for the joint tree. */
    std::string jointSplittingRule = "ANN_KD_SL_MIDPT";
    this->m_Configuration->ReadParameter( jointSplittingRule, "JointSplittingRule", 0, silentSplit );
		this->m_Configuration->ReadParameter( jointSplittingRule, "JointSplittingRule", level, true );

    /** Get the shrinking rule. */
    std::string shrinkingRule = "ANN_BD_SIMPLE";
    this->m_Configuration->ReadParameter( shrinkingRule, "ShrinkingRule", 0, silentShrink );
		this->m_Configuration->ReadParameter( shrinkingRule, "ShrinkingRule", level, true );

    /** Set the tree. */
    if ( treeType == "KDTree" )
    {
      /** If "SplittingRule" is given then all spiting rules are the same. */
      std::string tmp;
      if ( !this->m_Configuration->ReadParameter( tmp, "SplittingRule", 0, true ) )
      {
        this->SetANNkDTree( bucketSize, splittingRule );
      }
      else
      {
        this->SetANNkDTree( bucketSize, fixedSplittingRule, movingSplittingRule, jointSplittingRule );
      }
    }
    else if ( treeType == "BDTree" )
    {
      /** If "SplittingRule" is given then all spiting rules are the same. */
      std::string tmp;
      if ( !this->m_Configuration->ReadParameter( tmp, "SplittingRule", 0, true ) )
      {
        this->SetANNbdTree( bucketSize, splittingRule, shrinkingRule );
      }
      else
      {
        this->SetANNbdTree( bucketSize, fixedSplittingRule, movingSplittingRule,
          jointSplittingRule, shrinkingRule );
      }
      
    }
    else if ( treeType == "BruteForceTree" )
    {
      this->SetANNBruteForceTree();
    }

    /** Get the parameters for the search tree. */

    /** Get the tree search type. */
    std::string treeSearchType = "Standard";
    this->m_Configuration->ReadParameter( treeSearchType, "TreeSearchType", 0 );
		this->m_Configuration->ReadParameter( treeSearchType, "TreeSearchType", level, true );

    bool silentSR = true;
    if ( treeSearchType == "FixedRadius" )
    {
      silentSR = false;
    }

    /** Get the k nearest neighbours. */
		unsigned int kNearestNeighbours = 20;
    this->m_Configuration->ReadParameter( kNearestNeighbours, "KNearestNeighbours", 0 );
		this->m_Configuration->ReadParameter( kNearestNeighbours, "KNearestNeighbours", level, true );

    /** Get the error bound. */
    double errorBound = 0.0;
    this->m_Configuration->ReadParameter( errorBound, "ErrorBound", 0 );
		this->m_Configuration->ReadParameter( errorBound, "ErrorBound", level, true );

    /** Get the squared search radius. */
    double squaredSearchRadius = 0.0;
    this->m_Configuration->ReadParameter( squaredSearchRadius, "SquaredSearchRadius", 0, silentSR );
		this->m_Configuration->ReadParameter( squaredSearchRadius, "SquaredSearchRadius", level, true );
  
    /** Set the tree searcher. */
    if ( treeSearchType == "Standard" )
    {
      this->SetANNStandardTreeSearch( kNearestNeighbours, errorBound );
    }
    else if ( treeSearchType == "FixedRadius" )
    {
      this->SetANNFixedRadiusTreeSearch( kNearestNeighbours, errorBound, squaredSearchRadius );
    }
    else if ( treeSearchType == "Priority" )
    {
      this->SetANNPriorityTreeSearch( kNearestNeighbours, errorBound );
    }

	} // end BeforeEachResolution
	
  
} // end namespace elastix


#endif // end #ifndef __elxKNNGraphMultiDimensionalAlphaMutualInformationMetric_HXX__

