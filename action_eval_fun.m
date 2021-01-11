function results = action_eval_fun(testset, resdir, clsimgsetpath, competition)
    % copy this file into the VOCdevkit directory
    % change this path if you install the VOC code elsewhere
    addpath([cd '/VOCcode']);

    % initialize VOC options
    VOCinit;

    VOCopts.testset = testset;
    VOCopts.resdir = [resdir '/'];
    VOCopts.action.respath = [VOCopts.resdir '%s_action_' VOCopts.testset '_%s.txt'];
    VOCopts.action.clsimgsetpath = clsimgsetpath;
    
    results = [];
    sumap = 0;
    for i=2:VOCopts.nactions % skip other
        cls=VOCopts.actions{i};
        [recall,prec,ap]=VOCevalaction(VOCopts,competition,cls,false);
        results(i)=ap;
        sumap = sumap + ap;
    end
    map = sumap / 10;
    results(1)=map;
end