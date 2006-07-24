#ifndef __elxKNNGraphAlphaMutualInformationMetric_HXX__
#define __elxKNNGraphAlphaMutualInformationMetric_HXX__

#include "elxKNNGraphAlphaMutualInformationMetric.h"
#include "vnl/vnl_math.h"
#include <string>

namespace elastix
{
using namespace itk;

	/**
	 * ******************* Initialize ***********************
	 */

	template <class TElastix>
		void KNNGraphAlphaMutualInformationMetric<TElastix>
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
		void KNNGraphAlphaMutualInformationMetric<TElastix>
		::BeforeRegistration(void)
	{
    /** Get and set alpha, from alpha - MI. */
		double alpha = 0.5;
    this->m_Configuration->ReadParameter( alpha, "Alpha", 0 );
		this->SetAlpha( alpha );

  } // end BeforeRegistration

	/**
	 * ***************** BeforeEachResolution ***********************
	 */

	template <class TElastix>
		void KNNGraphAlphaMutualInformationMetric<TElastix>
		::BeforeEachResolution(void)
	{
		/** Get the current resolution level.*/
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

    /** Get the splitting rule. */
    std::string splittingRule = "ANN_KD_SL_MIDPT";
    this->m_Configuration->ReadParameter( splittingRule, "SplittingRule", 0, silentSplit );
		this->m_Configuration->ReadParameter( splittingRule, "SplittingRule", level, true );

    /** Get the shrinking rule. */
    std::string shrinkingRule = "ANN_BD_SIMPLE";
    this->m_Configuration->ReadParameter( shrinkingRule, "ShrinkingRule", 0, silentShrink );
		this->m_Configuration->ReadParameter( shrinkingRule, "ShrinkingRule", level, true );

    /** Set the tree. */
    if ( treeType == "KDTree" )
    {
      this->SetANNkDTree( bucketSize, splittingRule );
    }
    else if ( treeType == "BDTree" )
    {
      this->SetANNbdTree( bucketSize, splittingRule, shrinkingRule );
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


#endif // end #ifndef __elxKNNGraphAlphaMutualInformationMetric_HXX__

